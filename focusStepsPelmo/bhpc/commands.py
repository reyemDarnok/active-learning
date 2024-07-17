import contextlib
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE
from sys import stdin
from typing import Generator, Dict, List


class BHPCStateError(Exception):
    pass


@contextlib.contextmanager
def pushd(new_dir):
    """Emulates the behavior of pushd/popd.
    During the context the current working directory will be new_dir,
    and after closing the working directory will be restored to its old value.
    This contextmanager can be nested and overwrites working directory changes made inside it when exiting
    :param new_dir: The directory to move to during the context"""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


class BHPC:
    def __init__(self, request_auth_data_when_missing: bool = True, request_auth_data_when_invalid: bool = True,
                 bhpc_exe: Path = Path('C:\\_AWS', 'actualVersion', 'bhpc.exe'), auth_data: Dict = os.environ):
        self.ca_bundle: Path = Path('ca-certificates.crt')
        self.default_region: str = "eu-central-1"
        self.no_proxy: str = ".bayer.biz"
        self.proxy: str = "http://MVHNG:jA54QWMy@10.185.190.10:8080"
        self.session_token: str = None
        self.key: str = None
        self.key_id: str = None
        self.request_auth_data_when_missing = request_auth_data_when_missing
        self.request_auth_data_when_invalid = request_auth_data_when_invalid
        self.bhpc_exe = bhpc_exe
        self.read_auth_data_dict(auth_data)

    def read_auth_data_powershell(self, copy_paste: str):
        lines = copy_paste.splitlines()
        variables = {}
        for line in lines:
            if line.startswith("$ENV:"):
                kv_string = line[5]
                key = kv_string.split('=')[0]
                value = ''.join(kv_string.split('=')[1:])
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]
                variables[key] = value
        self.read_auth_data_dict(variables)

    def read_auth_data_dict(self, d: Dict[str, str]):
        if "AWS_ACCESS_KEY_ID" in d.keys():
            self.key_id = d["AWS_ACCESS_KEY_ID"]
        if "AWS_SECRET_ACCESS_KEY" in d.keys():
            self.key = d["AWS_SECRET_ACCESS_KEY"]
        if "AWS_SESSION_TOKEN" in d.keys():
            self.session_token = d["AWS_SESSION_TOKEN"]
        if "HTTPS_PROXY" in d.keys():
            self.proxy = d["HTTPS_PROXY"]
        if "NO_PROXY" in d.keys():
            self.no_proxy = d["NO_PROXY"]
        if "AWS_DEFAULT_REGION" in d.keys():
            self.default_region = d["AWS_DEFAULT_REGION"]
        if "AWS_CA_BUNDLE" in d.keys():
            self.ca_bundle = Path(d["AWS_DEFAULT_REGION"])

    def validate_auth_data(self, check_online: bool = True) -> bool:
        if self.key_id is None or self.key is None \
                or self.session_token is None \
                or self.proxy is None or self.no_proxy is None \
                or self.default_region is None or self.ca_bundle is None:
            return False
        if check_online:
            try:
                self.list()
            except:
                return False
        return True

    def request_auth_data(self):
        logger = logging.getLogger()
        logger.info("Requesting BHPC Credentials from user")
        print("To authorize this script against the BHPC, please copy the CLI Credentials from http://go/bhpc-prod")
        user_input = ""
        for line in stdin:
            line = line.strip()
            if line:
                user_input += line + '\n'
            else:
                break
        self.read_auth_data_powershell(copy_paste=user_input)
        if self.validate_auth_data(check_online=False):
            print("Thank you for providing the Credentials. Calls to the bhpc are now possible")
            logger.info("Received BHPC Credentials from user")
        else:
            logger.warning("BHPC Credentials by user were not valid")
            if input("Unfortunately there were missing fields in the Credentials. Try again? [Y/n]") != 'n':
                logger.debug("Retrying BHPC Credentials request")
                self.request_auth_data()

    def start_session(self, submit_folder: Path, submit_file_regex=r'.+\.sub',
                      session_name_prefix: str = "Unknown session", session_name_suffix: str = None,
                      machines: int = 1, cores: int = 2, multithreading: bool = True,
                      notification_email: str = None, session_timeout: int = 6) -> str:
        suffix = session_name_suffix if session_name_suffix else random.getrandbits(32)
        session = session_name_prefix + suffix
        logger = logging.getLogger()
        logger.debug('Starting session %s', session)
        logger.info('Starting upload of session %s', session)
        self.upload(submit_folder=submit_folder, submit_file_regex=submit_file_regex, session=session)
        logger.info('Starting run of session %s', session)
        self.run(session=session, machines=machines, cores=cores, multithreading=multithreading,
                 notification_email=notification_email, session_timeout=session_timeout)
        logger.debug('Session %s is now running on the BHPC', session)
        return session

    def upload(self, submit_folder: Path, session: str, submit_file_regex=r'.+\.sub'):
        self._execute_bhpc_command(
            [
                'upload',
                '-path', str(submit_folder),
                '-search', submit_file_regex,
                session
            ]
        )

    def run(self, session: str, machines: int = 1, cores: int = 2, multithreading: bool = False,
            notification_email: str = None, session_timeout: int = 6):
        assert cores in (2, 4, 8, 16, 96), f"Invalid core number {cores}. Only 2,4,8,16 or 96 are permitted"
        command_args = ['run',
                        '-force',
                        '-cores', str(cores),
                        '-count', str(machines),
                        session]
        if multithreading:
            command_args += ['-multi']
        if notification_email:
            command_args += ['-notificationEmail', notification_email]
        if session_timeout > 12:
            command_args += ['-longRun']

        run_process = self._execute_bhpc_command(command_args)
        if run_process.stdout.startswith('Session ') and run_process.stdout.endswith(' is not initialized.'):
            raise BHPCStateError(f"Session {session} could not be run because it was not initialized. "
                                 f"Initialize that session with the upload command first")

    def _execute_bhpc_command(self, arguments: List[str]) -> subprocess.CompletedProcess[str]:
        argv = [str(self.bhpc_exe.absolute()), *arguments]
        logger = logging.getLogger()
        logger.debug({"action": "Starting BHPC command", "arguments": argv})
        bhpc_process = subprocess.run(argv, text=True, capture_output=True, cwd=self.bhpc_exe.parent,
                                      encoding="windows-1252")
        if is_auth_message(bhpc_process.stdout):
            logger.info("BHPC rejected credentials, retrying after requesting new ones")
            self._handle_auth()
            return self._execute_bhpc_command(arguments)
        else:
            return bhpc_process



def download(session: str, wait_until_finished: bool = True, retry_interval: float = 60) -> bool:
    logger = logging.getLogger()
    try:
        if wait_until_finished:
            while not bhpc_job_finished(session):
                time.sleep(retry_interval)
        with pushd(bhpc_dir):
            logger.info('Running download command')
            download_process = subprocess.run([
                str(bhpc_exe.absolute()), 'download', session
            ], text=True, capture_output=True)
            logger.debug(download_process.stdout)
            if is_auth_message(download_process.stderr):
                request_auth_data()
                return download(session, wait_until_finished, retry_interval)
            return True
    except KeyboardInterrupt as e:
        if wait_until_finished:
            logger.warning(f"Download wait interrupted by interactive user")
            answer = input(
                f"Stopping monitoring session {session}. Should the session also be removed and killed? y/N:")
            if answer.strip().casefold() == "y".casefold():
                print("Removing session")
                logger.warning(f"Removing session {session} on request of interactive user")
                remove(session, True)
            raise e
        else:
            raise e


def remove(session: str, kill: bool = True):
    """Remove a session from the bhpc
    :param session: The session to remove
    :param kill: Whether to also kill the session"""
    logger = logging.getLogger()
    with pushd(bhpc_dir):
        logger.info('Running remove command')
        p = subprocess.Popen([
            str(bhpc_exe.absolute()), "remove", session],
            stdin=PIPE, stdout=PIPE, text=True)
        if kill:
            remove_stdout, _ = p.communicate("yes")
        else:
            remove_stdout, _ = p.communicate("no")
        logger.debug(remove_stdout)

        if is_auth_message(remove_stdout):
            request_auth_data()
            remove(session, kill)


def bhpc_job_finished(session: str) -> bool:
    """Checks whether the status message is a finished bhpc session
    :param session: The sessionID to check
    :return: True if all jobs in the session have finished status, False otherwise"""
    for status in get_bhpc_job_status(session):
        if not status.is_finished():
            return False
    return True


@dataclass
class Status:
    """A dataclass representing the status of a BHPC submit file"""
    initial: int
    """How many jobs are initialised, that is defined and uploaded but not yet running"""
    started: int
    """How many jobs are currently running"""
    done: int
    """How many jobs have already finished running"""
    submit_file: Path
    """The path of the submit file defining the jobs. 
    Any downloads from this job will be saved in the same directory as this file"""

    def is_finished(self):
        return self.initial == 0 and self.started == 0


def get_bhpc_job_status(session: str) -> Generator[Status, None, None]:
    """Checks the status of a BHPC job
    KNOWN ISSUE: Paths with whitespace other than single spaces will have their whitespace reduced to single spaces
    :param session: The session to check
    :return: The statuses of each submit file in the order BHPC show presents them"""
    logger = logging.getLogger()
    with pushd(bhpc_dir):
        logging.info('Running show command for session %s', session)
        p = subprocess.run([
            str(bhpc_exe.absolute()), 'show', session], text=True, capture_output=True)
    if is_auth_message(p.stdout):
        request_auth_data()
        yield from get_bhpc_job_status(session)
    logger.debug("%s", p.stdout)
    lines = p.stdout.splitlines()
    lines = lines[3:]
    for line in lines:
        if line.startswith('-'):
            return
        initial, started, done, *path = line.split()
        initial = int(initial)
        started = int(started)
        done = int(done)
        # attempt to repair path with spaces - only works if the only whitespace in path is single spaces, but that's
        # the most common case and this will most likely be used for reporting, not accessing the submit file
        path = Path(" ".join(path))
        logger.debug("%s %s %s %s", initial, started, done, path)
        yield Status(initial, started, done, path)


def is_auth_message(response: str) -> bool:
    """Returns True if response is the BHPC error message for missing authorization
    :param response: The response from the BHPC executable
    :return: True if response is the error response, False for all other values"""
    auth_message = ("--> Authorization environment variables not set. "
                    "Check if you have file with certificates in the same folder as the executable. "
                    "You will be redirected to http://go/bhpc-prod. "
                    "Please get variables and .crt File to use the bhpc cli <--\n")
    return auth_message == response
