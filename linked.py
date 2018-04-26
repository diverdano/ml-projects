#!usr/bin/env python


# === load libraries ===
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
        text.strip('Job Title\n')
        self.role, segment1 = text.split(' Company Name\n')
        self.co, segment2   = segment1.split('\nJob Location')
        self.other          = segment2.split('\n')
        self.location       = self.other.pop(0)
    def __repr__(self):
        return('role: {0}, co: {1}, location: {2}, other: {3}'.format(self.role, self.co, self.location, self.other))
    def __str__(self):
        return({
            'co'        : self.co,
            'role'      : self.role,
            'location'  : self.location,
            'other'     : self.other
        })

def getJobsTxt(file='/Users/dan/Downloads/linked.txt'):
    jobs = []
    ctas = []
    lines = util_data.readFile(file).split("\n\n")
    for line in lines:
        job = []
        cta = []
        if line[:9] == 'Job Title':
            line = line[9:]
            jobs.append(Job(line))
        elif line[:5] == 'Apply':
            cta = line.split('\n')
            ctas.append(cta)
        # job = (role, co, loc, note, when, cta)
        # jobs.append({'job': job, 'cta': cta})
    return (jobs)
