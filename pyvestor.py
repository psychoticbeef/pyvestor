#!/usr/bin/env python

import urllib
import urllib2
import os
import json as loli
import csv
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.optimize import fmin
from scipy.optimize import root
from scipy.optimize import least_squares
from scipy.optimize import leastsq
from numpy import linalg as LA
import pulp
from tabulate import tabulate
from collections import OrderedDict
import operator

# https://www.de.vanguard/web/cf/professionell/model.json?paths=[['getProfileData'],['layerContent','de'],[['labels','labelsByPath','portConfig'],'de','produktart,detailansicht'],['detailviewData','de','etf',9507,'equity','portfolio']]&method=get

# https://www.de.vanguard/web/cf/professionell/model.json?paths=%5B%5B%27getProfileData%27%5D%2C%5B%27layerContent%27%2C%27de%27%5D%2C%5B%5B%27labels%27%2C%27labelsByPath%27%2C%27portConfig%27%5D%2C%27de%27%2C%27produktart%2Cdetailansicht%27%5D%2C%5B%27detailviewData%27%2C%27de%27%2C%27etf%27%2C9524%2C%27equity%27%2C%27portfolio%27%5D%5D&method=get

#9522 9520 9507 9504 9523 9524

class Dist:
    def __init__(self, name):
        self.ratios = {}
        self.vector = []
        self.name = name
        self.regions = {}

    def getRatios(self):
        return self.ratios

    def verifySum(self):
        s = 0;
        for k, v in self.ratios.items():
            s = s + v
        return s

    def addRegion(self, name, pct):
        if not name in self.regions:
            self.regions[name] = 0.0
        self.regions[name] = self.regions[name] + pct

    def verifyRegions(self):
        total = 0
        for k, v in self.regions.items():
            total = total + v
        assert(total == 100)

    def addRatio(self, country, ratio):
        if country == "Other":
            return
        if country == "Korea":
            country = "South Korea"
        self.ratios[country] = ratio

    def vectorize(self, gdp):
        # init weights with zeroes
        self.vector = [0.0] * len(gdp)
        for k, v in self.ratios.items():
            self.vector[gdp.index(k)] = v
        return np.asarray(self.vector)

def make_dist(etf, out):
        result = Dist(etf)
        regions = {}
        with open(out, "r") as content_file:
            content = content_file.read()
            js = loli.loads(content)
            idx = js['jsonGraph']['detailviewData']['de']['etf'][str(etf)]['equity']['portfolio']['value']['countryExposure']
            for k in idx:
                countryName = k['countryName']
                if countryName == "Korea":
                    countryName = "South Korea"
                result.addRatio(countryName, k['fundMktPercent'])
                if not countryName == 'Other':
                    regionName = k['region']['regionName']
                    if regionName == 'Other' or regionName == 'Middle East':
                        regionName = 'Emerging Markets'
                    if not regionName in regions:
                        regions[regionName] = []
                    regions[regionName].append(countryName)
                    result.addRegion(regionName, k['fundMktPercent'])
        return result, regions

def download_dist(etf, out):
        domain = "https://www.de.vanguard/web/cf/professionell/model.json?"
        #url = "paths=[['getProfileData'],['layerContent','de'],[['labels','labelsByPath','portConfig'],'de','produktart,detailansicht'],['detailviewData','de','etf',{0},'equity','portfolio']]&method=get".format(etf)
        url = "paths=%5B%5B%27getProfileData%27%5D%2C%5B%27layerContent%27%2C%27de%27%5D%2C%5B%5B%27labels%27%2C%27labelsByPath%27%2C%27portConfig%27%5D%2C%27de%27%2C%27produktart%2Cdetailansicht%27%5D%2C%5B%27detailviewData%27%2C%27de%27%2C%27etf%27%2C{0}%2C%27equity%27%2C%27portfolio%27%5D%5D&method=get".format(etf)
        print(url)
        #url = urllib.quote(url)
        #print(url)
        response = urllib2.urlopen(domain + url)
        json = response.read()
        file = open(out, "w")
        file.write(json)
        file.close()

def read_gdp(infile):
    result = {}
    with open(infile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            result[row[0]] = int(row[1])
    return result

def fix_gdp(gdp, dists):
    gdpFixed = {}
    for etf, o in dists.items():
        for k, v in o.getRatios().items():
            if k in gdp:
                gdpFixed[k] = gdp[k]
    return gdpFixed

def percentage_gdp(gdp):
    result = {}
    totalGdp = float(sum(gdp.values()))
    for k, v in gdp.items():
        result[k] = v/totalGdp
    return result

def objective(x):
    r = np.zeros(len(vectorsHave[0]))
    for i in range(len(vectorsHave)):
        r = r + ((x[i]) * vectorsHave[i])
    r = r/100
    np.set_printoptions(suppress=True)
    return LA.norm((vectorWant - r), 1)**2

def constraint1(x):
    return x.sum() - 100

def constraint2(x):
    return (-x[0]-x[1]) + 35#-x[2]*0.1-

def mergeDicts(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            dict1[k] = dict1[k] + dict2[k]
        else:
            dict1[k] = dict2[k]

def uniqueDict(dict1):
    result = {}
    for k, v in dict1.items():
        result[k] = set(v)
    return result

def getRegionByCountry(regions, country_search):
    for region, countries in regions.items():
        for country in countries:
            if country == country_search:
                return region

def getDeveloped(regions):
    result = set()
    for region, countries in regions.items():
        if not region == 'Emerging Markets':
            result = result | countries
    return result

def downloadEtfData(etfs):
    allRegions = {}
    dists = {}
    for etf in etfs:
        out = "%s.json" % etf
        if not os.path.exists(out):
            download_dist(etf, out)
        dists[etf], regions = make_dist(etf, out)
        mergeDicts(allRegions, regions)
        if not abs(dists[etf].verifySum() - 100.0) <= 1e-09 :
            print("the sum of all countries is not 100pct! %s" % dists[etf].verifySum())
            exit()
    return dists, allRegions

def main():
    global vectorWant
    # can be gotten from the url
    etfs = OrderedDict([
            (9507, 'Emerging Markets'), 
            (666, 'Fake China Only'), 
            #(9505, 'All World'),
            (9522, 'Asia Pacific ex Japan'), 
            #(9520, 'Developed Europe'), 
            (9504, 'Japan'), 
            #(9523, 'North America'), 
            (9524, 'Developed Europe ex U.K.'), 
            (9527, 'Developed World'), 
            ])
    # manually extracted from wikipedia. too simple. only once / year.
    gdp = read_gdp("gdp.csv")

    # all regions (pacific, europe, emerging, ...)
    allRegions = {}
    # contains [etfId] => [country]:ratio
    dists, allRegions = downloadEtfData(etfs)
    #print("hi")
    allRegions = uniqueDict(allRegions)
    print tabulate(allRegions, headers=allRegions.keys())
    print("")

    # check if all countries are being found in gdp data. if not, there's a problem.
    # e.g. inconsistent naming (see "Korea" or "Other")
    for etf, o in dists.items():
        for k in o.getRatios():
             if not k in gdp:
                 print("%s not found %s" % (k, etf))
                 exit()

    # remove all countries we cannot invest in from gdp data
    gdpFixed = fix_gdp(gdp, dists)
    # calculate new gdp percentages ignoring countries we cannot invest in
    # about 9% in 2019, based on 2018 data
    gdpAdjusted = percentage_gdp(gdpFixed)
    #print tabulate(sorted([(round(v*100, 2),k) for k,v in gdpAdjusted.items()], reverse=True), headers=['GDP', 'Country'])
    print("")
    # calculate gdp per region
    gdpPerRegion = {}
    for region, countries in allRegions.items():
        gdpPerRegion[region] = 0.0
        for country in countries:
            gdpPerRegion[region] = gdpPerRegion[region] + gdpAdjusted[country]


    vectorSorted = []
    vectorWant = []
    # select an order for our items. we randomly picked size of gdp
    # everyone needs to stick to that
    for k in sorted( ((v,k) for k,v in gdpAdjusted.iteritems()), reverse=True):
        vectorSorted.append(k[1])
        vectorWant.append(k[0]*100)
        #print("%s\t\t\t%s" % (k[1], round(k[0]*100, 2)))
    # some stats
    totalGdp = sum(gdp.values())
    totalGdpFixed = sum(gdpFixed.values())
    gdpDeveloped = 0
    developed = getDeveloped(allRegions)
    for country in developed:
        gdpDeveloped = gdpDeveloped + gdpFixed[country]
    print tabulate([[totalGdp, totalGdpFixed, gdpDeveloped, round(float(totalGdpFixed)*100/totalGdp,2)]], headers=["Total GDP (m$)", "Adjusted GDP (m$)", "GDP Developed (m$)", "Market Percentage"])
    print("")

    # generate weights, ordered for all ETFs
    vectors = {}
    global vectorsHave
    vectorsHave = []
    for e in etfs:
        v = dists[e]
        distVectorized = v.vectorize(vectorSorted)
        vectors[k] = distVectorized
        vectorsHave.append(distVectorized)

    x0 = np.zeros(len(vectorsHave))
    for i in range(len(vectorsHave)):
        x0[i] = 100.0/len(vectorsHave)
    # fixme: initialize correct number
    b = (0.0, 100.0)
    bnds = (b,)*len(vectorsHave)
    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    cons = [con1]
    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
    res = np.zeros(len(vectorsHave[0]))
    for i in range(len(sol.x)):
        tmp = (sol.x[i]/100) * vectorsHave[i]
        #print("i: %s s: %s v: %s t: %s" % (i, sol.x[i], vectorsHave[i][1], tmp[1]))
        res = np.add(res, tmp)
    #print(res)
    #print("%s: %s" % (sol, sol.x.sum()))

    totalInvestedInDevelopedGdp = 0
    #for i in range(len(etfs)):
    #    etfs[i] = etfs[i] + (round(sol.x[i], 2))
        #print("%s %s" % (names[i].rjust(23), round(sol.x[i], 2)))
    i = 0
    result = []
    for k, v in etfs.items():
        result.append([round(sol.x[i], 2), k, v])
        i = i+1
    result.sort(key=operator.itemgetter(0), reverse=True)
    print tabulate(result, headers=["Percent", "etfId", "Name"])
    print("")

    investment = []
    investmentByRegion = {}
    for i in range(len(vectorSorted)):
        if vectorSorted[i] in developed:
            totalInvestedInDevelopedGdp = totalInvestedInDevelopedGdp + res[i]
        r = getRegionByCountry(allRegions, vectorSorted[i])
        if not r in investmentByRegion:
            investmentByRegion[r] = 0.0
        investmentByRegion[r] = investmentByRegion[r] + res[i]
        investment.append([vectorSorted[i], round(vectorWant[i],2), round(res[i],2), round(res[i]-vectorWant[i],2), round((res[i]/vectorWant[i])*100-100,2)])

    print tabulate(investment, headers=["Country", "GDP", "Invest", "Diff Abs", "Diff %"])
    print("")

    print tabulate(sorted([(round(v*100, 2),k,round(investmentByRegion[k],2),round(investmentByRegion[k]-v*100,2),(round(100*investmentByRegion[k]/(v*100)-100,2))) for k,v in gdpPerRegion.items()], reverse=True), headers=['GDP', 'Region', 'Invested', 'Diff Abs', 'Diff %'])
    print("")

    #print tabulate(sorted([(round(v*100, 2),k) for k,v in gdpAdjusted.items()], reverse=True), headers=['GDP', 'Country'])
    print tabulate([[res.sum(), round(totalInvestedInDevelopedGdp,2)]], headers=["Total invested (%)", "Invested in Developed (%)"])
    #for i in range(len(etfs)):
    #    print("%s %s" % (names[i], round(sol.x[i]*100, 2)))
    #for i in range(len(vectorSorted)):
    #    print("%s %s %s" % (vectorSorted[i], round(vectorWant[i],2), result[i]))


if __name__ == '__main__':
    main()

