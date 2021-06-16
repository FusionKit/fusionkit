import subprocess, sys

class Remote:
    '''
    Class to connect to remote clusters and execute remote bash shell commands
    '''
    def __init__(self):
        self.response = []

    def send_commands(self, bash='ssh',host=None,commands=None,verbose=True):
        remote = subprocess.Popen([bash, host, commands],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        self.response = remote.stdout.readlines()
        self.response = [line.decode("utf-8") for line in self.response]
        if self.response == []:
            error = remote.stderr.readlines()
            print("ERROR: {}".format(error), file=sys.stderr)
        else:
            if verbose:
                print(''.join(self.response))
            return self.response
    
