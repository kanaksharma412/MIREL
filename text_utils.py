import re

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords

# the set below has been updated arbitrarily
# please ignore any offensive words in this document

stopwords = set(stopwords.words('english'))
stopwords.update(
    ['a', 'able', 'about', 'above', 'abst', 'across', 'act', 'actually', 'added', 'adj', 'after', 'afterwards', 'again',
     'against', 'ah', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
     'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway',
     'anyways', 'anywhere', 'apparently', 'approximately', 'are', 'aren', 'arent', 'arise', 'around', 'as', 'aside',
     'ask', 'asking', 'at', 'auth', 'available', 'away', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes',
     'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being',
     'below', 'beside', 'besides', 'between', 'beyond', 'biol', 'both', 'brief', 'briefly', 'but', 'by', 'c', 'ca',
     'came', 'can', 'cannot', 'can\'t', 'cause', 'causes', 'certain', 'certainly', 'co', 'com', 'come', 'comes',
     'contain', 'containing', 'contains', 'could', 'couldnt', 'd', 'date', 'did', 'didn\'t', 'different', 'do', 'does',
     'doesn\'t', 'doing', 'done', 'don\'t', 'down', 'downwards', 'due', 'during', 'e', 'each', 'ed', 'edu', 'effect',
     'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'especially', 'et', 'et-al',
     'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'except', 'f', 'far',
     'few', 'ff', 'fifth', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'former', 'formerly',
     'forth', 'found', 'four', 'from', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given',
     'gives', 'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had', 'hardly', 'has', 'hasn\'t', 'have',
     'haven\'t', 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'heres', 'hereupon',
     'hers', 'herself', 'hes', 'hi', 'hid', 'him', 'himself', 'his', 'hither', 'home', 'how', 'howbeit', 'however',
     'hundred', 'i', 'id', 'ie', 'if', 'i\'ll', 'im', 'immediate', 'importance', 'important', 'in', 'inc', 'indeed',
     'index', 'information', 'instead', 'into', 'invention', 'inward', 'is', 'isn\'t', 'it', 'itd', 'it\'ll', 'its',
     'itself', 'i\'ve', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'kg', 'km', 'know', 'known', 'knows', 'l', 'largely',
     'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'lets', 'like', 'liked', 'likely',
     'line', 'little', '\'ll', 'look', 'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may',
     'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more',
     'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', 'na', 'name', 'namely',
     'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless',
     'new', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos',
     'not', 'noted', 'nothing', 'now', 'nowhere', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh',
     'ok', 'okay', 'old', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'ord', 'other', 'others',
     'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page',
     'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'poorly',
     'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'previously', 'primarily', 'probably',
     'promptly', 'proud', 'provides', 'put', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're',
     'readily', 'really', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related',
     'relatively', 'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 'run', 's', 'said', 'same',
     'saw', 'say', 'saying', 'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen',
     'self', 'selves', 'sent', 'seven', 'several', 'shall', 'she', 'shed', 'she\'ll', 'shes', 'should', 'shouldn\'t',
     'show', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 'similar', 'similarly', 'since',
     'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime',
     'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying',
     'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup',
     'sure', 't', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that',
     'that\'ll', 'thats', 'that\'ve', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
     'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'there\'ll', 'thereof', 'therere', 'theres', 'thereto',
     'thereupon', 'there\'ve', 'these', 'they', 'theyd', 'they\'ll', 'theyre', 'they\'ve', 'think', 'this', 'those',
     'thou', 'though', 'thoughh', 'thousand', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to',
     'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two',
     'u', 'un', 'under', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'us', 'use', 'used',
     'useful', 'usefully', 'usefulness', 'uses', 'using', 'v', 'value', 'various', '\'ve', 'very', 'via', 'viz', 'vol',
     'vols', 'vs', 'w', 'want', 'was', 'wasnt', 'way', 'we', 'wed', 'we\'ll', 'went', 'were', 'werent', 'will',
     'we\'ve', 'what', 'whatever', 'what\'ll', 'whats', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
     'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 'who',
     'whod', 'whoever', 'whole', 'who\'ll', 'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'with', 'within',
     'without', 'wont', 'words', 'world', 'would', 'wouldnt', 'www', 'x', 'y', 'yes', 'yet', 'you', 'youd', 'you\'ll',
     'your', 'youre', 'yours', 'yourself', 'yourselves', 'you\'ve', 'z', 'zero'])
stopwords.update(
    ['ain', 'ain\'t', 'aint', 'verse', 'chorus', 'huh', 'hah', 'thing', 'yeah', 'yuh', 'yup', 'ya', 'yall', 'dont',
     'mayne' 'vocal', 'vocals', 'woah', 'whoa', 'ooh', 'oh', 'oom', 'uh', 'joe', 'demi', 'brock', 'murphy', 'harris',
     'zappa', 'bozzio', 'going', 'coming', 'shoulda', 'woulda', 'hey', 'hoo', 'yer', 'work', 'job', 'bag', 'doo',
     'sayin', 'tha', 'mmm', 'hmm', 'bee', 'whatcha', 'wha', 'mon', 'doot', 'mow', 'cha', 'aye', 'kno', 'uhh', 'ima',
     'knew', 'rollin', 'la', 'rah', 'mek', 'nuh', 'dem', 'wit', 'choo', 'chu', 'weh', 'deh', 'wah', 'waan', 'inna',
     'hee', 'naw', 'ohh', 'ahh', 'aah'])

df_w2e = pd.read_csv('word_to_emotion.csv')
df_w2e = df_w2e[df_w2e['sum'] != 0]
words_w2e = [value for idx, value in df_w2e['word'].iteritems()]
stopwords.difference_update(words_w2e)
df_w2e = df_w2e[df_w2e['sum'] == 0]
words_w2e = [value for idx, value in df_w2e['word'].iteritems()]
stopwords.update(words_w2e)

contractions_dict = {
    " yea ": " yeah ",
    " i mma ": " i am going to ",
    " imma ": " i am going to ",
    " em ": " them ",
    " dat ": " that ",
    " gimme ": " give me ",
    " niggas ": " nigga ",
    " niggaz ": " nigga ",
    " cuz ": " cause ",
    " tryna ": " trying to ",
    " tryin ": " trying ",
    " lookin ": " looking ",
    " gon ": " going to ",
    " goin ": " going to ",
    " gonna ": " going to ",
    " gotta ": " got to ",
    " outta ": " out of ",
    " wanna ": " want to ",
    " bout ": " about ",
    " bein ": " being ",
    " lil ": " little ",
    " lovin ": " loving ",
    " doin ": " doing ",
    " nothin ": " nothing ",
    " livin ": " living ",
    " killin ": " killing ",
    " fuckin ": " fucking ",
    " thang ": " thing ",
    " askin ": " asking ",
    " frontin ": " fronting ",
    " jus ": " just ",
    " neva ": " never ",
    " nuttin ": " nothing ",
    " ova ": " over ",
    " playin ": " playing ",
    " tellin ": " telling ",
    " trippin ": " tripping ",
    " gal ": " girl ",
    " momma ": " mother ",
    " mommy ": " mother ",
    " mom ": " mother ",
    " mama ": " mother ",
    " papa ": " father ",
    " dad ": " father ",
    " daddy ": " father ",
    " dada ": " father ",
    " alright ": " all right ",
    " aight ": " all right "
}


def remove_words(text:str, set_of_words):
    """
    Removes any words present in the text that are included in the set_of_words

    """
    words = text.strip().split()
    words = [word for word in words if word not in set_of_words]
    return ' '.join(words)


def remove_proper_nouns(text):
    """
    Removes proper nouns from the tagged text, and returns untagged text.
    :param text:
    :return:
    """
    return ' '.join([word for word, pos in pos_tag(text.strip().split()) if pos != 'NNP' and pos != 'NNPS'])


def sub(find:str, replacement:str, text:str):
    """
    A wrapper over re.sub.
    Created initially to remove replace complete words, not as extensively used by the time the project ended.

    :param find: the string to find
    :param replacement: the replacement string
    :param text: the text to search within
    :return: updated text
    """
    pat = re.compile(r"([^\w])(" + find + r")([^\w])")
    return re.sub(pat, r'\1' + replacement + r'\3', text)


def expand_contractions(text: str):
    """
    Expands the contractions present in the text, using the contractions dictionary.

    :return: text:str
    """
    for contraction, expanded in contractions_dict.items():
        text = text.replace(contraction, expanded)
    return text


def clean_text(text: str):
    """
    Cleans the text, removing any non-word characters, removes proper nouns, makes the text lowercase,
    expands contractions and removes stopwords

    :param text: The text to clean
    :return: text:
    """
    text = text.replace('’', '\'')
    # text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    text = re.sub('(\w+)in\'( |\n|$)', r'\1' + r'ing' + r'\2', text)
    text = re.sub('[^\w]', ' ', text)
    text = remove_proper_nouns(text)
    text = " " + text.lower() + " "
    text = sub("x*[\d]|[\d]x*", ' ', text)
    text = expand_contractions(text)
    text = remove_words(text, stopwords)
    text = re.sub('[ ]+', ' ', text)
    return text.strip()
