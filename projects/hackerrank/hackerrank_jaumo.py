#!/bin/env python3
def countHosts(filename):
    hosts   = {}
    result  = []
    for record in data:
        host, message = record.split(' ', 1) # split on the 1st space, host is the first value, message is rest of log record
        if host in hosts.keys(): hosts[host] +=1 # if host has already been added, increment the counter
        else:                  hosts[host] = 1 # otherwise add it to the dictionary
    print(hosts)
    for host in hosts.keys():
        result.append(host + ' ' + str(hosts[host]))  # append the hostname and count of hosts (from dictionary counter)
    return result

if __name__ == '__main__':
    filename = input()
    print('filename is: {}'.format(filename))
    with open(filename, 'r') as infile:
        data=infile.read().strip().split('\n')              # strip final new line
    print(data)

    result = countHosts(data)
    fileout = "records_" + filename             # prepend "records_" to log file name
    fptr = open(fileout, 'w')
    for record in result:
        fptr.writeline(str(result) + '\n')
    fptr.close()
