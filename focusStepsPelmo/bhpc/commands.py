import contextlib
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import PIPE
from sys import stdin
from typing import Generator, Dict, List, Tuple, Optional

from focusStepsPelmo.util.datastructures import TypeCorrecting


class BHPCStateError(Exception):
    pass


@dataclass(frozen=True)
class SubmitFileStatus(TypeCorrecting):
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


@dataclass(frozen=True)
class SessionStatus(TypeCorrecting):
    submit_files: List[SubmitFileStatus]

    @staticmethod
    def from_bhpc_message(bhpc_message: str) -> 'SessionStatus':
        raise NotImplementedError()


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


@dataclass
class SessionSummary(TypeCorrecting):
    session_id: str
    cwid: Optional[str]
    status: str  # TODO make Enum
    instance_type: Optional[str]  # TODO make Enum
    vCPUs: Optional[int]
    creation_time: Optional[datetime]
    elapsed_time: Optional[timedelta]
    initialized: Optional[int]
    started: Optional[int]
    finished: Optional[int]

    def __post_init__(self):
        if self.initialized in ('N/A', '-'):
            self.initialized = None
        if self.started in ('N/A', '-'):
            self.started = None
        if self.finished in ('N/A', '-'):
            self.finished = None
        if self.elapsed_time:
            parts = self.elapsed_time.split(':')
            self.elapsed_time = timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


@dataclass(frozen=True)
class BHPCState(TypeCorrecting):
    sessions: List[SessionSummary]
    active_sessions: int

    @staticmethod
    def from_bhpc_message(bhpc_message: str):
        section = 0
        for line in bhpc_message.splitlines():
            if section == 0:
                if line.startswith('|'):
                    section = 1
            elif section == 1:
                section = 2
            elif section

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
            if input(
                    "Unfortunately there were missing fields in the Credentials. Try again? [Y/n]").strip().casefold() != 'n'.strip().casefold():
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
        # TODO Error handling session already exists
        # TODO Error handling no submit files found

    def run(self, session: str, machines: int = 1, cores: int = 2, multithreading: bool = False,
            notification_email: str = None, session_timeout: int = 6):
        assert cores in (2, 4, 8, 16, 96), f"Invalid core number {cores}. Only 2,4,8,16 or 96 are permitted"
        assert machines > 0, "The number of machines must be a positive number or the job will stall"
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
        if run_process[0].startswith('Session ') and run_process[0].endswith(' is not initialized.'):
            raise BHPCStateError(f"Session {session} could not be run because it was not initialized. "
                                 f"Initialize that session with the upload command first")


    def download(self, session: str, wait_until_finished: bool = True,
                 retry_interval: timedelta = timedelta(seconds=60)) -> bool:
        logger = logging.getLogger()
        try:
            if wait_until_finished:
                before_last_check = datetime.now()
                while not self.is_session_finished(session):
                    after_last_check = datetime.now()
                    sleep_interval = retry_interval - (after_last_check - before_last_check)
                    logger.debug('Sleeping for %s before the next check of session status', sleep_interval)
                    time.sleep(sleep_interval.total_seconds() + sleep_interval.microseconds / 1_000_000)
                    before_last_check = datetime.now()
                logger.info('Finished wait for the completion of %s', session)
                self._execute_bhpc_command(['download', session])
                return True
                # TODO report missing files if not waiting
        except KeyboardInterrupt as e:
            if wait_until_finished:
                logger.warning("Download wait interrupted by interactive user")
                answer = input(
                    f"Stopping monitoring session {session}. Should the session also be removed and killed? y/N:")
                if answer.strip().casefold() == 'y'.casefold():
                    print("Removing session")
                    logger.warning(f"Removing session {session} on request of interactive user")
                    self.remove(session, True)
                raise e
            else:
                raise e

    def remove(self, session: str, kill: bool = True):
        logger = logging.getLogger()
        logger.info('Running remove command for session %s', session)
        self._execute_bhpc_command(["remove", session], 'yes' if kill else 'no')

    def list(self) -> BHPCState:
        raise NotImplementedError()

    def show(self, session) -> SessionStatus:
        return self.get_session_status(session)

    def is_session_finished(self, session: str):
        for status in self.get_session_status(session).submit_files:
            if not status.is_finished:
                return False
        return True

    def get_session_status(self, session) -> SessionStatus:
        stdout, _ = self._execute_bhpc_command(["show", session])
        return SessionStatus.from_bhpc_message(stdout)

    def _execute_bhpc_command(self, arguments: List[str], stdin: str = "") -> Tuple[str, str]:
        argv = [str(self.bhpc_exe.absolute()), *arguments]
        env = os.environ.copy()

        logger = logging.getLogger()
        logger.debug({"action": "Starting BHPC command", "arguments": argv})
        bhpc_process = subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=self.bhpc_exe.parent, env=env,
                                        text=True, encoding="windows-1252")
        stdout, stderr = bhpc_process.communicate(input=stdin)
        if is_auth_message(stdout):
            logger.info("BHPC rejected credentials, retrying after requesting new ones")
            self._handle_auth()
            return self._execute_bhpc_command(arguments)
        else:
            return stdout, stderr


def get_bhpc_job_status(session: str) -> Generator[SubmitFileStatus, None, None]:
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
        yield SubmitFileStatus(initial, started, done, path)


def is_auth_message(response: str) -> bool:
    """Returns True if response is the BHPC error message for missing authorization
    :param response: The response from the BHPC executable
    :return: True if response is the error response, False for all other values"""
    auth_message = ("--> Authorization environment variables not set. "
                    "Check if you have file with certificates in the same folder as the executable. "
                    "You will be redirected to http://go/bhpc-prod. "
                    "Please get variables and .crt File to use the bhpc cli <--\n")
    return auth_message == response
