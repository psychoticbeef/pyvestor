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
    fixed_gdp = {}
    for etf, o in dists.items():
        for k, v in o.getRatios().items():
            if k in gdp:
                fixed_gdp[k] = gdp[k]
    return fixed_gdp

def percentage_gdp(gdp):
    result = {}
    total_gdp = float(sum(gdp.values()))
    for k, v in gdp.items():
        result[k] = v/total_gdp
    return result

def objective(x):
    r = np.zeros(len(vectors_have[0]))
    for i in range(len(vectors_have)):
        r = r + ((x[i]) * vectors_have[i])
    r = r/100
    np.set_printoptions(suppress=True)
    return LA.norm((vector_want - r), 1)

# sum equals 1
def constraint1(x):
    return x.sum() - 100

def constraint2(x):
    return (-x[2]-x[7]*0.1) + 35

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

def main():
    global vector_want
    developed = [
            "Canada",
            "Italy",
            "France",
            "Ireland",
            "Norway",
            "Israel",
            "Australia",
            "Singapore",
            "Germany",
            "Belgium",
            "Hong Kong",
            "Spain",
            "Netherlands",
            "Denmark",
            "Poland",
            "Finland",
            "United States",
            "Sweden",
            "Switzerland",
            "New Zealand",
            "Portugal",
            "United Kingdom",
            "Austria",
            "Japan",
            "South Korea"
            ]
    # can be gotten from the url
    etfs = OrderedDict([
            (9522, 'Asia Pacific ex Japan'), 
            (9520, 'Developed Europe'), 
            (9507, 'Emerging Markets'), 
            (9504, 'Japan'), 
            (9523, 'North America'), 
            (9524, 'Developed Europe ex U.K.'), 
            (9527, 'Developed World'), 
            (9505, 'All World')
            ])
    #print tabulate(sorted([(v,k) for k,v in etfs.items()]), headers=['Name', 'etfId'])
    #print("")

    # manually extracted from wikipedia. too simple. only once / year.
    gdp = read_gdp("gdp.csv")

    # all regions (pacific, europe, emerging, ...)
    all_regions = {}
    # contains [etfId] => [country]:ratio
    dists = {}
    # download / read ETF data
    for etf in etfs:
        out = "%s.json" % etf
        if not os.path.exists(out):
            download_dist(etf, out)
        dists[etf], regions = make_dist(etf, out)
        mergeDicts(all_regions, regions)
        #print(dists[etf].getRatios())
        if not abs(dists[etf].verifySum() - 100.0) <= 1e-09 :
            print("the sum of all countries is not 100pct! %s" % dists[etf].verifySum())
            exit()
    #print("hi")
    all_regions = uniqueDict(all_regions)
    print tabulate(all_regions, headers=all_regions.keys())
    print("")

    # check if all countries are being found in gdp data. if not, there's a problem.
    # e.g. inconsistent naming (see "Korea" or "Other")
    for etf, o in dists.items():
        for k in o.getRatios():
             if not k in gdp:
                 print("%s not found %s" % (k, etf))
                 exit()

    # remove all countries we cannot invest in from gdp data
    fixed_gdp = fix_gdp(gdp, dists)
    # calculate new gdp percentages ignoring countries we cannot invest in
    # about 9% in 2019, based on 2018 data
    adjusted_gdp = percentage_gdp(fixed_gdp)
    #print tabulate(sorted([(round(v*100, 2),k) for k,v in adjusted_gdp.items()], reverse=True), headers=['GDP', 'Country'])
    print("")
    # calculate gdp per region
    gdp_per_region = {}
    for region, countries in all_regions.items():
        gdp_per_region[region] = 0.0
        for country in countries:
            gdp_per_region[region] = gdp_per_region[region] + adjusted_gdp[country]


    sorted_vector = []
    vector_want = []
    # select an order for our items. we randomly picked size of gdp
    # everyone needs to stick to that
    for k in sorted( ((v,k) for k,v in adjusted_gdp.iteritems()), reverse=True):
        sorted_vector.append(k[1])
        vector_want.append(k[0]*100)
        #print("%s\t\t\t%s" % (k[1], round(k[0]*100, 2)))
    # some stats
    total_gdp = sum(gdp.values())
    total_fixed_gdp = sum(fixed_gdp.values())
    gdp_developed = 0
    for country in developed:
        gdp_developed = gdp_developed + fixed_gdp[country]
    print tabulate([[total_gdp, total_fixed_gdp, gdp_developed, round(float(total_fixed_gdp)*100/total_gdp,2)]], headers=["Total GDP (m$)", "Adjusted GDP (m$)", "GDP Developed (m$)", "Market Percentage"])
    print("")

    # generate weights, ordered for all ETFs
    vectors = {}
    global vectors_have
    vectors_have = []
    #for k, v in dists.items():
    #    dist_vectorized = v.vectorize(sorted_vector)
    #    vectors[k] = dist_vectorized
    #    vectors_have.append(dist_vectorized)
    for e in etfs:
        v = dists[e]
        dist_vectorized = v.vectorize(sorted_vector)
        vectors[k] = dist_vectorized
        vectors_have.append(dist_vectorized)

    x0 = np.zeros(len(vectors_have))
    for i in range(len(vectors_have)):
        x0[i] = 100.0/len(vectors_have)
    # fixme: initialize correct number
    b = (0.0, 100.0)
    bnds = (b,)*len(vectors_have)
    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    cons = [con1,con2]
    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
    res = np.zeros(len(vectors_have[0]))
    for i in range(len(sol.x)):
        tmp = (sol.x[i]/100) * vectors_have[i]
        #print("i: %s s: %s v: %s t: %s" % (i, sol.x[i], vectors_have[i][1], tmp[1]))
        res = np.add(res, tmp)
    #print(res)
    #print("%s: %s" % (sol, sol.x.sum()))

    invest_dev_gdp = 0
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
    investment_by_region = {}
    for i in range(len(sorted_vector)):
        if sorted_vector[i] in developed:
            invest_dev_gdp = invest_dev_gdp + res[i]
        r = getRegionByCountry(all_regions, sorted_vector[i])
        if not r in investment_by_region:
            investment_by_region[r] = 0.0
        investment_by_region[r] = investment_by_region[r] + res[i]
        investment.append([sorted_vector[i], round(vector_want[i],2), round(res[i],2), round(res[i]-vector_want[i],2), round((res[i]/vector_want[i])*100-100,2)])

    print tabulate(investment, headers=["Country", "GDP", "Invest", "Diff Abs", "Diff %"])
    print("")

    print tabulate(sorted([(round(v*100, 2),k,round(investment_by_region[k],2),round(investment_by_region[k]-v*100,2),(round(100*investment_by_region[k]/(v*100)-100,2))) for k,v in gdp_per_region.items()], reverse=True), headers=['GDP', 'Region', 'Invested', 'Diff Abs', 'Diff %'])
    print("")

    #print tabulate(sorted([(round(v*100, 2),k) for k,v in adjusted_gdp.items()], reverse=True), headers=['GDP', 'Country'])
    print tabulate([[res.sum(), round(invest_dev_gdp,2)]], headers=["Total invested (%)", "Invested in Developed (%)"])
    #for i in range(len(etfs)):
    #    print("%s %s" % (names[i], round(sol.x[i]*100, 2)))
    #for i in range(len(sorted_vector)):
    #    print("%s %s %s" % (sorted_vector[i], round(vector_want[i],2), result[i]))


if __name__ == '__main__':
    main()

