from PyQt6 import QtGui, QtCore
from PyQt6.QtGui import QColorConstants

class XMLHighlighter(QtGui.QSyntaxHighlighter):
 
    #INIT THE STUFF
    def __init__(self, parent=None):
        super(XMLHighlighter, self).__init__(parent)
 
        keywordFormat = QtGui.QTextCharFormat()
        keywordFormat.setForeground(QColorConstants.DarkMagenta)
        keywordFormat.setFontWeight(QtGui.QFont.Weight.Bold)
 
        keywordPatterns = ["\\b?xml\\b", "/>", ">", "<"]
 
        self.highlightingRules = [(QtCore.QRegularExpression(pattern), keywordFormat)
                for pattern in keywordPatterns]
 
        xmlElementFormat = QtGui.QTextCharFormat()
        xmlElementFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        xmlElementFormat.setForeground(QColorConstants.Green)
        self.highlightingRules.append((QtCore.QRegularExpression("\\b[A-Za-z0-9_]+(?=[\s/>])"), xmlElementFormat))
 
        xmlAttributeFormat = QtGui.QTextCharFormat()
        xmlAttributeFormat.setFontItalic(True)
        xmlAttributeFormat.setForeground(QColorConstants.Blue)
        self.highlightingRules.append((QtCore.QRegularExpression("\\b[A-Za-z0-9_]+(?=\\=)"), xmlAttributeFormat))
 
        self.valueFormat = QtGui.QTextCharFormat()
        self.valueFormat.setForeground(QColorConstants.Red)
 
        self.valueStartExpression = QtCore.QRegularExpression("\"")
        self.valueEndExpression = QtCore.QRegularExpression("\"(?=[\s></])")
 
        singleLineCommentFormat = QtGui.QTextCharFormat()
        singleLineCommentFormat.setForeground(QColorConstants.Gray)
        self.highlightingRules.append((QtCore.QRegularExpression("<!--[^\n]*-->"), singleLineCommentFormat))
 
    #VIRTUAL FUNCTION WE OVERRIDE THAT DOES ALL THE COLLORING
    def highlightBlock(self, text):
 
        #for every pattern
        for pattern, format in self.highlightingRules:
 
            #Create a regular expression from the retrieved pattern
            expression = QtCore.QRegularExpression(pattern)

            i = expression.globalMatch(text)
            while i.hasNext():
                match = i.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

          
 
        #    #Check what index that expression occurs at with the ENTIRE text
        #    index = expression.indexIn(text)
 #
        #    #While the index is greater than 0
        #    while index >= 0:
 #
        #        #Get the length of how long the expression is true, set the format from the start to the length with the text format
        #        length = expression.matchedLength()
        #        self.setFormat(index, length, format)
 #
        #        #Set index to where the expression ends in the text
        #        index = expression.indexIn(text, index + length)
 #
        ##HANDLE QUOTATION MARKS NOW.. WE WANT TO START WITH " AND END WITH ".. A THIRD " SHOULD NOT CAUSE THE WORDS INBETWEEN SECOND AND THIRD TO BE COLORED
        #self.setCurrentBlockState(0)
 #
        #startIndex = 0
        #if self.previousBlockState() != 1:
        #    startIndex = self.valueStartExpression.indexIn(text)
 #
        #while startIndex >= 0:
        #    endIndex = self.valueEndExpression.indexIn(text, startIndex)
 #
        #    if endIndex == -1:
        #        self.setCurrentBlockState(1)
        #        commentLength = len(text) - startIndex
        #    else:
        #        commentLength = endIndex - startIndex + self.valueEndExpression.matchedLength()
 #
        #    self.setFormat(startIndex, commentLength, self.valueFormat)
 #
        #    startIndex = self.valueStartExpression.indexIn(text, startIndex + commentLength)