import subprocess 
import sys, os

class Remote:
    '''
    Class to connect to remote clusters and execute remote bash shell commands
    '''
    def __init__(self):
        self.response = []

    def send_commands(self, bash='ssh',host=None,commands=None,verbose=True):
        # Ports are handled in ~/.ssh/config since we use OpenSSH
        print('Executing remote commands over {} on {}...'.format(bash,host))
        remote = subprocess.Popen([bash, host, commands],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        self.response = remote.stdout.readlines()
        self.response = [line.decode("utf-8") for line in self.response]
        if self.response == []:
            error = remote.stderr.readlines()
            error = [line.decode("utf-8") for line in error]
            if error != []:
                print("ERROR: {}".format(error), file=sys.stderr)
        else:
            if verbose:
                print(''.join(self.response))
            return self.response
    
    def copy_file_to_remote(origin=None,destination=None):
        try:
            print('Moving file to: {}'.format(destination))
            process = subprocess.Popen(['scp {} {}'.format(origin,destination)],shell=True)
            sts = os.waitpid(process.pid, 0)
        except CalledProcessError:
            print('ERROR: Connection to host failed!')