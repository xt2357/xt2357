# coding=utf8
import re
import nltk.data
# from nltk.tokenize.treebank import TreebankWordTokenizer
# from nltk.corpus import stopwords
import jieba


nltk_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize


def split_into_sentences_by_nltk(text):
    sentences = nltk_sentence_tokenizer(text)
    assert len(sentences) > 0, u'split result: %s' % sentences
    return sentences


# english_punctuations = \
#     {u',', u'.', u':', u';', u'?', u'(', u')', u'[', u']', u'&', u'!', u'*', u'@', u'#', u'$', u'%', u'-'}
# english_stopwords = set(stopwords.words(u'english'))
# word_tokenizer = TreebankWordTokenizer().tokenize


def tokenize(text):
    return [tok for tok in jieba.cut(text) if tok.isalnum()]


sentence_pattern = re.compile(ur'.*[0-9A-Za-z]+.*', re.UNICODE)


def split_into_paragraph_sentence_token(text):
    """
    split the whole text into [[[[word], [word], ..], [sentence], ..], [paragraph], ..]
    treat every non-empty line as a paragraph
    :rtype: [[[[word], [word], ..], [sentence], ..], [paragraph], ..]
    """
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
