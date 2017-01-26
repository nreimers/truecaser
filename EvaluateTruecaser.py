from Truecaser import *
import cPickle
import nltk
import string


def evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    correctTokens = 0
    totalTokens = 0
    
    for sentence in testSentences:
        tokensCorrect = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokensCorrect]
        tokensTrueCase = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        
        perfectMatch = True
        
        for idx in xrange(len(tokensCorrect)):
            totalTokens += 1
            if tokensCorrect[idx] == tokensTrueCase[idx]:
                correctTokens += 1
            else:
                perfectMatch = False
        
        if not perfectMatch:
            print tokensCorrect
            print tokensTrueCase
        
            print "-------------------"
    

    print "Accuracy: %.2f%%" % (correctTokens / float(totalTokens)*100)
    
    
def defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    testSentences = [
        "Its website was launched on February 4, 2004 by Mark Zuckerberg with his Harvard College roommates and fellow students Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes."
        ,"Facebook is a for-profit corporation and online social networking service based in Menlo Park, California, United States. "
        ,"The founders had initially limited the website's membership to Harvard students, but later expanded it to colleges in the Boston area, the Ivy League, and Stanford University. "
        ,"It gradually added support for students at various other universities and later to high school students. "
        ,"Since 2006, anyone in general aged 13 and older has been allowed to become a registered user of the website, though variations exist in the minimum age requirement, depending on applicable local laws."
        ,"Its name comes from the face book directories often given to American university students."
        ,"Because of the large volume of data that users submit to the service, Facebook has come under scrutiny for their privacy policies. Facebook, Inc. held its initial public offering in February 2012 and began selling stock to the public three months later, reaching an original peak market capitalization of $104 billion."
        ,"Zuckerberg wrote a program called Facemash on October 28, 2003 while attending Harvard University as a sophomore (second year student)."
        ,"Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services."
        ,"Its hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, and the Apple Watch smartwatch."
        ,"Apple's consumer software includes the OS X and iOS operating systems, the iTunes media player, the Safari web browser, and the iLife and iWork creativity and productivity suites."
        ,"Its online services include the iTunes Store, the iOS App Store and Mac App Store, and iCloud."
        ,"Microsoft Corporation (commonly referred to as Microsoft) is an American multinational technology company headquartered in Redmond, Washington, that develops, manufactures, licenses, supports and sells computer software, consumer electronics and personal computers and services."
        ,"Its best known software products are the Microsoft Windows line of operating systems, Microsoft Office office suite, and Internet Explorer and Edge web browsers."
        ,"Its flagship hardware products are the Xbox game consoles and the Microsoft Surface tablet lineup."
        ,"It is the world's largest software maker by revenue, and one of the world's most valuable companies."
        ,"Google is an American multinational technology company specializing in Internet-related services and products."
        ,"These include online advertising technologies, search, cloud computing, and software."
        ,"Most of its profits are derived from AdWords, an online advertising service that places advertising near the list of search results."
        ,"Rapid growth since incorporation has triggered a chain of products, acquisitions and partnerships beyond Google's core search engine (Google Search)."
        ,"It offers online productivity software (Google Docs) including email (Gmail), a cloud storage service (Google Drive) and a social networking service (Google+)."
        ,"Desktop products include applications for web browsing (Google Chrome), organizing and editing photos (Google Photos), and instant messaging and video chat (Hangouts)."
        ,"The company leads the development of the Android mobile operating system and the browser-only Chrome OS for a class of netbooks known as Chromebooks and desktop PCs known as Chromeboxes."
        ,"Google has moved increasingly into communications hardware, partnering with major electronics manufacturers[20] in the production of its \"high-quality low-cost\" Nexus devices."
        ,"In 2012, a fiber-optic infrastructure was installed in Kansas City to facilitate a Google Fiber broadband service."
        ,"WhatsApp Messenger is a proprietary cross-platform, encrypted, instant messaging client for smartphones."
        ,"It uses the Internet to send text messages, documents, images, video, user location and audio messages to other users using standard cellular mobile numbers."
        ,"As of February 2016, WhatsApp had a user base of one billion, making it the most popular messaging application."
        ,"WhatsApp Inc., based in Mountain View, California, United States, was acquired by Facebook Inc. on February 19, 2014, for approximately US$19.3 billion"
        ,"Barack Hussein Obama II (born August 4, 1961) is an American politician serving as the 44th President of the United States."
        ,"He is the first African American to hold the office, as well as the first president born outside of the continental United States."
        ,"Born in Honolulu, Hawaii, Obama is a graduate of Columbia University and Harvard Law School, where he served as president of the Harvard Law Review."
        ,"He was a community organizer in Chicago before earning his law degree."
        ,"He worked as a civil rights attorney and taught constitutional law at University of Chicago Law School between 1992 and 2004."
        ,"He served three terms representing the 13th District in the Illinois Senate from 1997 to 2004, and ran unsuccessfully in the Democratic primary for the United States House of Representatives in 2000 against incumbent Bobby Rush."
        ,"In 2004, Obama received national attention during his campaign to represent Illinois in the United States Senate with his victory in the March Democratic Party primary, his keynote address at the Democratic National Convention in July, and his election to the Senate in November."
        ,"He began his presidential campaign in 2007 and, after a close primary campaign against Hillary Clinton in 2008, he won sufficient delegates in the Democratic Party primaries to receive the presidential nomination."
        ,"He then defeated Republican nominee John McCain in the general election, and was inaugurated as president on January 20, 2009."
        ,"Nine months after his inauguration, Obama was named the 2009 Nobel Peace Prize laureate."
        ,"Albert Einstein was a German-born theoretical physicist. He developed the general theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics)."
        ,"Einstein's work is also known for its influence on the philosophy of science."
        ,"Einstein is best known in popular culture for his mass-energy equivalence formula E = mc2 (which has been dubbed \"the world's most famous equation\")."
        ,"He received the 1921 Nobel Prize in Physics for his \"services to theoretical physics\", in particular his discovery of the law of the photoelectric effect, a pivotal step in the evolution of quantum theory."
        ,"Near the beginning of his career, Einstein thought that Newtonian mechanics was no longer enough to reconcile the laws of classical mechanics with the laws of the electromagnetic field."
        ,"This led to the development of his special theory of relativity."
        ,"He realized, however, that the principle of relativity could also be extended to gravitational fields, and with his subsequent theory of gravitation in 1916, he published a paper on general relativity."
        ,"He continued to deal with problems of statistical mechanics and quantum theory, which led to his explanations of particle theory and the motion of molecules. He also investigated the thermal properties of light which laid the foundation of the photon theory of light."
        ,"In 1917, Einstein applied the general theory of relativity to model the large-scale structure of the universe."
        ,"Ulm is a city in the federal German state of Baden-Wuerttemberg, situated on the River Danube."
        ,"The city, whose population is estimated at almost 120,000 (2015), forms an urban district of its own and is the administrative seat of the Alb-Donau district."
        ,"Ulm, founded around 850, is rich in history and traditions as a former Free Imperial City."
        ,"Today, it is an economic centre due to its varied industries, and it is the seat of the University of Ulm."
        ,"Internationally, Ulm is primarily known for having the church with the tallest steeple in the world (161.53 m or 529.95 ft), the Gothic minster (Ulm Minster) and as the birthplace of Albert Einstein."
    ]
    
    evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    
if __name__ == "__main__":       
    f = open('english_distributions.obj', 'rb')
    uniDist = cPickle.load(f)
    backwardBiDist = cPickle.load(f)
    forwardBiDist = cPickle.load(f)
    trigramDist = cPickle.load(f)
    wordCasingLookup = cPickle.load(f)
    f.close()
    
    defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
