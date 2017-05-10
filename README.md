# PyTorch Marvelous ChatBot

![PicName](http://ofwzcunzi.bkt.clouddn.com/EItTIwqpcrAsrexq.png)


> Aim to build a Marvelous ChatBot


# Synopsis

This is the first and the only opensource of **ChatBot**, I will continues update this repo personally, aim to build a intelligent ChatBot, as the next version of Jarvis.

This repo will maintain to build a **Marvelous ChatBot** based on PyTorch,
welcome star and submit PR.

![PicName](http://ofwzcunzi.bkt.clouddn.com/FmLv0IpsmiMkLGiQ.png)
# Already Done

Currently this repo did those work:

* based on official tutorial, this repo will move on develop a seq2seq chatbot, QA system;
* re-constructed whole project, separate mess code into `data`, `model`, `train logic`;
* model can be save into local, and reload from previous saved dir, which is lack in official tutorial;
* just replace the dataset you can train your own data!

Last but not least, this project will maintain or move on other repo in the future but we will
continue implement a practical seq2seq based project to build anything you want: **Translate Machine**,
**ChatBot**, **QA System**... anything you want.


# Requirements

```
PyTorch
python3
Ubuntu Any Version
Both CPU and GPU can works
```

# Usage

Before dive into this repo, you want to glance the whole structure, we have these setups:

* `config`: contains the config params, which is global in this project, you can change a global param here;
* `datasets`: contains data and data_loader, using your own dataset, you should implement your own data_loader but just a liitle change on this one;
* `models`: contains seq2seq model definition;
* `utils`: this folder is very helpful, it contains some code may help you get out of anoying things, such as save model, or catch KeyboardInterrupt exception or load previous model, all can be done in here.

to train model is also straightforward, just type:
```
python3 train.py
```

# Contribute

wecome submit PR!!!! Let's build ChatBot together!

# Contact

if you have anyquestion, you can find me via wechat `jintianiloveu`
