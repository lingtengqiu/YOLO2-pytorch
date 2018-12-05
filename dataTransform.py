# coding:utf-8
# !/usr/bin/env python
import os
import os.path
from lxml import etree, objectify
import cv2
import shutil


class dataTransform:
    def __init__(self):
        self.currentPath = os.getcwd()
        self.classnames = {0: 'defect'}
        self.classnamesToValue = {'defect': 0}

    def singXmlToTxt(self, xmlFilePath, txtFilePath):
        xml = etree.parse(xmlFilePath)  # 读取test.xml文件
        root = xml.getroot()  # 获取根节点
        for node in root.getchildren():
            with open(txtFilePath, 'a') as txtFile:
                if node.tag == ("object"):
                    line = ''
                    objectes = node.getchildren()
                    for object in objectes:
                        if object.tag == 'name':
                            line = line + str(self.classnamesToValue[object.text]) + ' '
                        if (object.tag == 'bndbox'):
                            bndboxes = object.getchildren()
                            for bndbox in bndboxes:
                                if bndbox.tag == 'ymax':
                                    line = line + bndbox.text + '\n'
                                else:
                                    line = line + bndbox.text + ' '

                    txtFile.write(line)

    def xmlToTxt(self, xmlFolderName, txtFolderName):
        xmlFolderPath = os.path.join(self.currentPath, xmlFolderName)
        txtFolderPath = os.path.join(self.currentPath, txtFolderName)
        if not os.access(txtFolderPath, os.F_OK):
            os.mkdir(txtFolderPath)
        xmlFileList = os.listdir(xmlFolderPath)
        for xmlFile in xmlFileList:
            xmlFilePath = os.path.join(xmlFolderPath, xmlFile)
            txtFilePath = os.path.join(txtFolderPath, xmlFile[:-3] + 'txt')
            self.singXmlToTxt(xmlFilePath, txtFilePath)

    def addObject(self, className, xmin, ymin, xmax, ymax, E, anno_tree):
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(className),
            E.pose('Unspecified'),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmax),
                E.ymax(ymax)
            ),
        )
        anno_tree.append(anno_tree2)

    def getImageInformation(self, imagePath):
        image = cv2.imread(imagePath)
        shape = image.shape
        if len(shape) == 2:
            return shape[0], shape[1], 1
        else:
            return (shape[0], shape[1], shape[2])

    def getImageWH(self, imagePath):
        image = cv2.imread(imagePath)
        shape = image.shape
        return (shape[0], shape[1])

    def writeXml(self, file_name, save_where, img_info,classnames,ind_info, bnd_info):
        if not os.path.exists(save_where):
            os.mkdir(save_where)
        self.classnames = classnames
        pathDetail = file_name.strip().split('/')
        E = objectify.ElementMaker(annotate=False)
        xmlFilePath = os.path.join(save_where, pathDetail[-1][:-3] + 'xml')
        width, height,_ = img_info.shape
        depth = 3
        print(xmlFilePath)
        anno_tree = E.annotation(
            E.folder(pathDetail[-2]),
            E.filename(pathDetail[-1]),
            E.source(
                E.database('qlt'),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(depth)
            ),
            E.segmented(0),
        )
        for ind, information in zip(ind_info,bnd_info):
            self.addObject(self.classnames[int(ind)], information[0], information[1],
                           information[2],
                           information[3], E, anno_tree)
        etree.ElementTree(anno_tree).write(xmlFilePath, pretty_print=True)

    def txtToXml(self, txtFolderName, imageFolderName, xmlFolderName, pictureExtent):
        txtFolder = os.path.join(self.currentPath, txtFolderName)
        xmlFolder = os.path.join(self.currentPath, xmlFolderName)
        if not os.access(xmlFolder, os.F_OK):
            os.mkdir(xmlFolder)
        fileList = os.listdir(txtFolder)
        for file in fileList:
            filePath = os.path.join(txtFolder, file)
            self.writeXml(filePath, imageFolderName, xmlFolderName, pictureExtent)

    def singFileNormalize(self, txtFilePath, imageFilePath, normalFilePath):
        print(imageFilePath)
        W, H = self.getImageWH(imageFilePath)
        W = float(W)
        H = float(H)
        dw = 1. / W
        dh = 1. / H
        # print(targetFile)
        with open(txtFilePath) as fileS:
            with open(normalFilePath, 'w') as fileT:
                line = fileS.readline()
                while line:
                    information = line.strip().split(' ')
                    xmin = float(information[1])
                    ymin = float(information[2])
                    xmax = float(information[3])
                    ymax = float(information[4])
                    width = xmax - xmin
                    height = ymax - ymin
                    cx = xmin + 0.5 * width
                    cy = ymin + 0.5 * height
                    cxNorm = cx * dw
                    cyNorm = cy * dh
                    widthNorm = width * dw
                    heightNorm = height * dh
                    lineTemp = information[0] + ' ' + str(cxNorm) + ' ' + str(cyNorm) + ' ' + str(
                        widthNorm) + ' ' + str(heightNorm) + '\n'
                    fileT.write(lineTemp)
                    line = fileS.readline()

    def txtNormalization(self, txtFolder, imagesFolder, normalizationFolder, pictureExtent):
        txtPath = os.path.join(self.currentPath, txtFolder)
        imagesPath = os.path.join(self.currentPath, imagesFolder)
        normalPath = os.path.join(self.currentPath, normalizationFolder)
        if not os.access(normalPath, os.F_OK):
            os.mkdir(normalPath)
        fileList = os.listdir(txtPath)
        for file in fileList:
            txtFilePath = os.path.join(txtPath, file)
            imageFilePath = os.path.join(imagesPath, file[:-3] + pictureExtent)
            normalFilePath = os.path.join(normalPath, file)
            self.singFileNormalize(txtFilePath, imageFilePath, normalFilePath)

    def singleXmlToTxt(self, xmlFilePath, txtFilePath):
        xml = etree.parse(xmlFilePath)  # 读取test.xml文件
        root = xml.getroot()  # 获取根节点
        W = 0.0
        H = 0.0

        cx = 0.0
        cy = 0.0
        width = 0.0
        height = 0.0
        for node in root.getchildren():
            if node.tag == "size":
                size = node.getchildren()
                for WH in size:
                    if WH.tag == 'width':
                        W = float(WH.text)
                    if WH.tag == 'height':
                        H = float(WH.text)
        dw = 1. / W
        dh = 1. / H
        for node in root.getchildren():

            with open(txtFilePath, 'a') as txtFile:
                if node.tag == ("object"):
                    line = ''
                    objectes = node.getchildren()
                    for object in objectes:
                        if object.tag == 'name':
                            line = line + str(self.classnamesToValue[object.text]) + ' '
                        if (object.tag == 'bndbox'):
                            bndboxes = object.getchildren()

                            xmin, ymin, xmax, ymax = 0, 0, 0, 0
                            for bndbox in bndboxes:
                                if bndbox.tag == 'xmin':
                                    xmin = float(bndbox.text)
                                if bndbox.tag == 'xmax':
                                    xmax = float(bndbox.text)
                                if bndbox.tag == 'ymin':
                                    ymin = float(bndbox.text)
                                if bndbox.tag == 'ymax':
                                    ymax = float(bndbox.text)

                            width = xmax - xmin
                            height = ymax - ymin
                            cx = (xmin + 0.5 * width) * dw
                            cy = (ymin + 0.5 * height) * dh

                            widhtNormal = width * dw
                            heightNormal = height * dh
                            line = line + str(cx) + ' ' + str(cy) + ' ' + str(widhtNormal) + ' ' + str(
                                heightNormal) + '\n'
                    txtFile.write(line)

    def xmlToNormal(self, xmlFolderName, normalFolderName):
        xmlFolderPath = os.path.join(self.currentPath, xmlFolderName)
        normalFolderPath = os.path.join(self.currentPath, normalFolderName)
        if not os.access(normalFolderPath, os.F_OK):
            os.mkdir(normalFolderPath)
        xmlFileList = os.listdir(xmlFolderPath)
        for xmlFile in xmlFileList:
            xmlFilePath = os.path.join(xmlFolderPath, xmlFile)
            normalFilePath = os.path.join(normalFolderPath, xmlFile[:-3] + 'txt')
            self.singleXmlToTxt(xmlFilePath, normalFilePath)


def main():
    operate = dataTransform()
    # operate.txtToXml('labels','images','xml','png')
    # operate.txtNormalization('labels','images','normalization','jpg')
    # operate.xmlToTxt('xml',"labels")
    operate.xmlToNormal('xml', 'normalization')


if __name__ == '__main__':
    main()