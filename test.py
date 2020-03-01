# author: Paul Galatic
#
# Program to

# STD LIB
import logging
import argparse

# EXTERNAL LIB

# LOCAL LIB

# CONSTANTS
LOGFILE = 'bootstrap.log'
LOGFORMAT = '%(asctime)s %(name)s:%(levelname)s -- %(message)s'

def parse_args():
    '''Parses arguments'''
    ap = argparse.ArgumentParser()

    
    
    return ap.parse_args()
    
def main():
    '''Driver program'''
    args = parse_args()
    logging.basicConfig(filename=LOGFILE, filemode='a', format=LOGFORMAT, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('\n-----START-----')

    

    logging.info('\n------END-----\n')

if __name__ == '__main__':
    main()