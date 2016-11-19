# coding=utf8
import re
import nltk.data
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
import jieba

caps = u"([A-Z])"
prefixes = u"(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = u"(Inc|Ltd|Jr|Sr|Co)"
starters = u"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = u"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = u"[.](com|net|org|io|gov)"
digits = u"([0-9])"


def split_into_sentences(text):
    text = text.strip()
    # ensure at least one sentence is returned
    assert len(text) > 0
    if text[-1].isalnum():
        text += u'.'
    text = u" " + text + u"  "
    text = text.replace(u"\n", u" ")
    text = re.sub(prefixes, u"\\1<prd>", text)
    text = re.sub(websites, u"<prd>\\1", text)
    text = re.sub(digits + u"[.]" + digits, u"\\1<prd>\\2", text)
    if u"Ph.D" in text:
        text = text.replace(u"Ph.D.", u"Ph<prd>D<prd>")
    text = re.sub(u"\s" + caps + u"[.] ", u" \\1<prd> ", text)
    text = re.sub(acronyms + u" " + starters, u"\\1<stop> \\2", text)
    text = re.sub(caps + u"[.]" + caps + u"[.]" + caps + u"[.]", u"\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + u"[.]" + caps + u"[.]", u"\\1<prd>\\2<prd>", text)
    text = re.sub(u" " + suffixes + u"[.] " + starters, u" \\1<stop> \\2", text)
    text = re.sub(u" " + suffixes + u"[.]", u" \\1<prd>", text)
    text = re.sub(u" " + caps + u"[.]", u" \\1<prd>", text)
    if u"”" in text:
        text = text.replace(u".”", u"”.")
    if u"\"" in text:
        text = text.replace(u".\"", u"\".")
    if u"!" in text:
        text = text.replace(u"!\"", u"\"!")
    if u"?" in text:
        text = text.replace(u"?\"", u"\"?")
    text = text.replace(u".", u".<stop>")
    text = text.replace(u"?", u"?<stop>")
    text = text.replace(u"!", u"!<stop>")
    text = text.replace(u"<prd>", u".")
    sentences = text.split(u"<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


nltk_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize


def split_into_sentences_by_nltk(text):
    sentences = nltk_sentence_tokenizer(text)
    assert len(sentences) > 0, u'split result: %s' % sentences
    return sentences


english_punctuations = \
    {u',', u'.', u':', u';', u'?', u'(', u')', u'[', u']', u'&', u'!', u'*', u'@', u'#', u'$', u'%', u'-'}
english_stopwords = set(stopwords.words(u'english'))
word_tokenizer = TreebankWordTokenizer().tokenize


def tokenize(text):
    return [tok for tok in jieba.cut(text) if tok.isalnum()]


sentence_pattern = re.compile(ur'.*[0-9A-Za-z]+.*', re.UNICODE)


def split_into_paragraph_sentence_token(text):
    result = \
        [[tokenize(sentence) for sentence in split_into_sentences_by_nltk(paragraph)
          if sentence_pattern.match(sentence)]
         for paragraph in text.splitlines() if paragraph.strip() != u'']
    result = [[sent for sent in para if len(sent) > 0] for para in result]
    return [para for para in result if len(para) > 0]


TEXT = ur'''TRENTON, N.J. — The Wal-Mart truck driver from Georgia accused of triggering a crash in New Jersey that critically injured Tracy Morgan and killed another comedian had not slept for more than 24 hours.
The information about 35-year-old Kevin Roper’s lack of sleep is contained in a criminal complaint.
A court administrator says Roper will face an initial court appearance on Wednesday.
Roper has been charged with death by auto and four counts of assault by auto.
Authorities say Roper apparently failed to slow for traffic early Saturday and swerved to avoid a crash. His rig smashed into the back of Morgan’s Mercedes limo bus, killing comedian James “Jimmy Mack” McNair.'''


def main():
    # print ([token for token in tokens])
    print (tokenize(TEXT))
    print (split_into_sentences_by_nltk(TEXT))
    print ([tokenize(sentence) for sentence in split_into_sentences_by_nltk(TEXT)])
    print (split_into_paragraph_sentence_token(TEXT))
    print (u'a\nb\n\nc'.splitlines())
    print (split_into_sentences_by_nltk(u'Mr. wang is good.'))


if __name__ == '__main__':
    main()
