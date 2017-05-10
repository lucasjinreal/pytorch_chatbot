import math
import random
import re
import time
import unicodedata
from io import open
import torch
from torch.autograd import Variable
import torch.optim as optim
from models.models import *
from utils.model_utils import *
from datasets.data_loader import PairDataLoader


def train_model(data_loader, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion,
                max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    try:
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
    except KeyboardInterrupt:
        return

    decoder_input = Variable(torch.LongTensor([[data_loader.SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        try:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                loss += criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing
        except KeyboardInterrupt:
            return

    else:
        # Without teacher forcing: use its own predictions as the next input
        try:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output[0], target_variable[di])
                if ni == data_loader.EOS_token:
                    break
        except KeyboardInterrupt:
            return

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train(data_loader, encoder, decoder, n_epochs, print_every=100, save_every=1000, evaluate_every=100,
          learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    encoder, decoder, start_epoch = load_previous_model(encoder, decoder, CHECKPOINT_DIR, MODEL_PREFIX)

    for epoch in range(start_epoch, n_epochs + 1):

        input_variable, target_variable = data_loader.get_pair_variable()

        try:
            loss = train_model(data_loader, input_variable, target_variable, encoder,
                               decoder, encoder_optimizer, decoder_optimizer, criterion)
        except KeyboardInterrupt:
            pass
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % save_every == 0:
            save_model(encoder, decoder, CHECKPOINT_DIR, MODEL_PREFIX, epoch)

        if epoch % evaluate_every == 0:
            evaluate_randomly(data_loader, encoder, decoder, n=1)


def evaluate(data_loader, encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = data_loader.variable_from_sentence(data_loader.input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[data_loader.SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == data_loader.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(data_loader.output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(data_loader, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(data_loader.pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(data_loader, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def main():

    pair_data_loader = PairDataLoader()
    hidden_size = 256
    encoder1 = EncoderRNN(pair_data_loader.input_lang.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, pair_data_loader.output_lang.n_words,
                                   1, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()
    print('start training...')
    pair_data_loader.get_pair_variable()
    train(pair_data_loader, encoder1, attn_decoder1, 75000)
    evaluate_randomly(pair_data_loader, encoder1, attn_decoder1)


if __name__ == '__main__':
    main()
