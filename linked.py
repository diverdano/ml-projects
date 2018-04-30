#!usr/bin/env python


# === load libraries ===
import util
import util_data
import html

# === test functions ===

def getJobsHTML(file='/Users/dan/Downloads/LinkedIn.html'):
    # util_data.sniffDelim(file)
    jobs = ""
    lines = util_data.readFile(file).split()
    fourth = sorted([len(line) for line in lines], reverse=True)[3]
    longlines = [line for line in lines if len(line) > fourth]
    for line in longlines:
        if "Senior" in line: jobs = line

    return html.unescape(longlines), html.unescape(jobs)

class Job(object):
    '''structure for job object'''
    role        = ''
    co          = ''
    location    = ''
    other       = ''
    def __init__(self, text):
        self.parseLine(text)
        # self.titleExcept()
    def parseLine(self,text):
        self.role, segment1 = text.split(' Company Name\n')
        self.co, segment2   = segment1.split('\nJob Location')
        self.other          = segment2.split('\n')
        self.location       = self.other.pop(0)
    def titleExcept(self):
        self.role           = util.titleExcept(self.role)
        self.co             = util.titleExcept(self.co)
        self.location       = util.titleExcept(self.location)
        self.other          = [util.titleExcept(item) for item in self.other]
    def __repr__(self):
        return('role: {0}, co: {1}, location: {2}, other: {3}'.format(self.role, self.co, self.location, self.other))
    def __str__(self):
        return('{0}, {1}, {2}, '.format(self.role, self.co, self.location) + ", ".join(self.other))
    def __str2__(self):
        return({
            'co'        : self.co,
            'role'      : self.role,
            'location'  : self.location,
            'other'     : self.other
        })
    def __str3__(self):
        return([self.role, self.co, self.location] + self.other)

def getJobsTxt(file='/Users/dan/Downloads/linked.txt'):
    jobs = []
    ctas = []
    lines = util_data.readFile(file).split("\n\n")
    for line in lines:
        job = []
        cta = []
        if line[:10] == 'Job Title\n':
            line = line[10:]
            jobs.append(Job(line))
        elif line[:5] == 'Apply':
            cta = line.split('\n')
            ctas.append(cta)
        # job = (role, co, loc, note, when, cta)
        # jobs.append({'job': job, 'cta': cta})
    return (jobs)
