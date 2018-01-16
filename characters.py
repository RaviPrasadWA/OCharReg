# -*- encoding: utf-8 -*-

import re

digits                  = u"0123456789"
letters                 = u"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
symbols                 = ur"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
ascii                   = digits+letters+symbols
xsymbols                = u"""€¢£»«›‹÷©®†‡°∙•◦‣¶§÷¡¿▪▫"""
german                  = u"ÄäÖöÜüß"
french                  = u"ÀàÂâÆæÇçÉéÈèÊêËëÎîÏïÔôŒœÙùÛûÜüŸÿ"
turkish                 = u"ĞğŞşıſ"
greek                   = u"ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
portuguese              = u"ÁÃÌÍÒÓÕÚáãìíòóõú"
telugu                  = u" ఁంఃఅఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహఽాిీుూృౄెేైొోౌ్ౘౙౠౡౢౣ౦౧౨౩౪౫౬౭౮౯"
default                 = ascii+xsymbols+german+french+portuguese
european                = default+turkish+greek

replacements            = [
                            (u'[_~#]',u"~"),
                            (u'"',u"''"),
                            (u"`",u"'"),
                            (u'[“”]',u"''"),
                            (u"´",u"'"),
                            (u"[‘’]",u"'"),
                            (u"[“”]",u"''"),
                            (u"“",u"''"),
                            (u"„",u",,"),
                            (u"…",u"..."),
                            (u"′",u"'"),
                            (u"″",u"''"),
                            (u"‴",u"'''"),
                            (u"〃",u"''"),
                            (u"µ",u"μ"),
                            (u"[–—]",u"-"),
                            (u"ﬂ",u"fl"),
                            (u"ﬁ",u"fi"),
                            (u"ﬀ",u"ff"),
                            (u"ﬃ",u"ffi"),
                            (u"ﬄ",u"ffl"),
                        ]
def requote(s):
    s                   = unicode(s)
    s                   = re.sub(ur"''",u'"',s)
    return s

def requote_fancy(s,germanic=0):
    s                   = unicode(s)
    if germanic:
        s               = re.sub(ur"\s+''",u"”",s)
        s               = re.sub(u"''\s+",u"“",s)
        s               = re.sub(ur"\s+,,",u"„",s)
        s               = re.sub(ur"\s+'",u"’",s)
        s               = re.sub(ur"'\s+",u"‘",s)
        s               = re.sub(ur"\s+,",u"‚",s)
    else:
        s               = re.sub(ur"\s+''",u"“",s)
        s               = re.sub(ur"''\s+",u"”",s)
        s               = re.sub(ur"\s+,,",u"„",s)
        s               = re.sub(ur"\s+'",u"‘",s)
        s               = re.sub(ur"'\s+",u"’",s)
        s               = re.sub(ur"\s+,",u"‚",s)
    return s