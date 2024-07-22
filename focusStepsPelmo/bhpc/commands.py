"""A file describing the commands that can be sent to the BHPC"""
import contextlib
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from subprocess import PIPE
from sys import stdin
from typing import Dict, List, Tuple, Optional

from focusStepsPelmo.util.datastructures import TypeCorrecting


class BHPCStateError(Exception):
    """An Error indicating that the request is incompatible with the current state of the BHPC"""
    pass


class BHPCAccessError(Exception):
    """An Error indicating that this script failed to access the BHPC"""
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

    def is_finished(self) -> bool:
        """True if all jobs have finished, false otherwise"""
        return self.initial == 0 and self.started == 0


@dataclass(frozen=True)
class SessionStatus(TypeCorrecting):
    """A dataclass describing a single session"""
    submit_files: List[SubmitFileStatus]
    """A list of the statuses of the .sub files making up this session"""

    @staticmethod
    def from_bhpc_message(bhpc_message: str) -> 'SessionStatus':
        """Parse a message from the BHPC into a SessionStatus object
        :param bhpc_message: The message to parse
        :return: An object describing the session described by the bhpc_message"""
        lines = bhpc_message.splitlines()
        lines = lines[3:]  # remove headings
        submit_files = []
        for line in lines:
            if line.startswith('-'):
                break
            initial, started, done, path = line.split(maxsplit=3)
            initial = int(initial)
            started = int(started)
            done = int(done)
            path = Path(path)
            submit_files.append(SubmitFileStatus(initial, started, done, path))
        return SessionStatus(submit_files)
    # TODO other run information in last section of show


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
    """A summary of all session data, as returned by the BHPC list command"""
    session_id: str
    """The id of the session"""
    cwid: Optional[str]
    """Who started the run of the session, if anybody"""
    status: str  # TODO make Enum
    """Which state the session is in right now"""
    instance_type: Optional[str]  # TODO make Enum
    """On what type of amazon resources the session is deployed, if any"""
    vCPUs: Optional[int]
    """How many vCPUs are assigned to the session, if it has started to run"""
    creation_time: Optional[datetime]
    """When was the session started"""
    elapsed_time: Optional[timedelta]
    """How long did the session run, if it has been started"""
    initialized: Optional[int]
    """If the session is running or has finished, how many jobs are in initialized state"""
    started: Optional[int]
    """If the session is running or has finished, how many jobs are in started state"""
    finished: Optional[int]
    """If the session is running or has finished, how many jobs are in finished state"""

    def __post_init__(self):
        if self.initialized in ('N/A', '-'):
            self.initialized = None
        if self.started in ('N/A', '-'):
            self.started = None
        if self.finished in ('N/A', '-'):
            self.finished = None
        if self.elapsed_time and type(self.elapsed_time) == str:
            # still in init, type hints are not yet correct
            # noinspection PyUnresolvedReferences
            parts = self.elapsed_time.split(':')
            self.elapsed_time = timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))


class BHPCListSections(int, Enum):
    """Identifiers for different Sections in the BHPC list response"""
    PREAMBLE = 0
    TABLE_HEADLINE = 1
    TABLE_ENTRIES = 2
    ACTIVE_SESSION_COUNT = 3


@dataclass(frozen=True)
class BHPCState(TypeCorrecting):
    """A description of the state of the BHPC, as returned by the list call"""
    sessions: List[SessionSummary]
    active_sessions: int

    @classmethod
    def from_bhpc_message(cls, bhpc_message: str) -> 'BHPCState':
        """Parse a bhpc message from a string into a BHPCState object"""
        section = BHPCListSections.PREAMBLE
        sessions = []
        for line in bhpc_message.splitlines():
            if section == BHPCListSections.PREAMBLE:
                if line.startswith('|'):
                    section = BHPCListSections.TABLE_HEADLINE
            elif section == BHPCListSections.TABLE_HEADLINE:
                section = BHPCListSections.TABLE_ENTRIES
            elif section == BHPCListSections.TABLE_ENTRIES:
                if line.startswith('|'):
                    parts = [x.strip() for x in line.split('|')]
                    sessions.append(SessionSummary(*parts[1:-1]))
                else:
                    section = BHPCListSections.ACTIVE_SESSION_COUNT
            elif section == BHPCListSections.ACTIVE_SESSION_COUNT:
                if line:
                    return cls(sessions, int(line.split()[0]))
        raise ValueError("BHPC Message was malformed. Was the input taken from the BHPC list command?")


class BHPC:
    """A connector to the BHPC. Manages credentials but relies on the bhpc exe being installed"""

    def __init__(self, request_auth_data_when_missing: bool = True, request_auth_data_when_invalid: bool = True,
                 bhpc_exe: Path = Path('C:\\_AWS', 'actualVersion', 'bhpc.exe'), auth_data: Dict = os.environ):
        self.ca_bundle: Path = Path('ca-certificates.crt')
        self.default_region: str = "eu-central-1"
        self.no_proxy: str = ".bayer.biz"
        self.proxy: str = "http://MVHNG:jA54QWMy@10.185.190.10:8080"
        self.session_token: Optional[str] = None
        self.key: Optional[str] = None
        self.key_id: Optional[str] = None
        self.request_auth_data_when_missing = request_auth_data_when_missing
        self.request_auth_data_when_invalid = request_auth_data_when_invalid
        self.bhpc_exe = bhpc_exe
        self.read_auth_data_dict(auth_data)

    def read_auth_data_powershell(self, copy_paste: str):
        logger = logging.getLogger()
        lines = copy_paste.splitlines()
        variables = {}
        for line in lines:
            logger.debug({"status": "listing lines", "line": line})
            if line.startswith("$Env:"):
                kv_string = line.split(':', maxsplit=1)[1]
                key = kv_string.split('=')[0].strip()
                value = ''.join(kv_string.split('=')[1:]).strip()
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]
                variables[key] = value
        logger.debug({"status": "read env vars from user in powershell format", "vars": variables})
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
            self.ca_bundle = Path(d["AWS_CA_BUNDLE"])

    def _get_bhpc_env(self) -> Dict[str, str]:
        env = {
            "HTTPS_PROXY": self.proxy,
            "NO_PROXY": self.no_proxy,
            "AWS_DEFAULT_REGION": self.default_region,
            "AWS_CA_BUNDLE": str(self.ca_bundle)
        }
        if self.key_id:
            env["AWS_ACCESS_KEY_ID"] = self.key_id
        if self.key:
            env["AWS_SECRET_ACCESS_KEY"] = self.key
        if self.session_token:
            env["AWS_SESSION_TOKEN"] = self.session_token
        return env

    def validate_auth_data(self, check_online: bool = True) -> bool:
        if self.key_id is None or self.key is None \
                or self.session_token is None \
                or self.proxy is None or self.no_proxy is None \
                or self.default_region is None or self.ca_bundle is None:
            return False
        if check_online:
            try:
                self.list()
            except BHPCAccessError:
                return False
        return True

    def request_auth_data(self) -> bool:
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
            return True
        else:
            logger.warning("BHPC Credentials by user were not valid")
            if (input("Unfortunately there were missing fields in the Credentials. Try again? [Y/n]").strip().casefold()
                    != 'n'.strip().casefold()):
                logger.debug("Retrying BHPC Credentials request")
                return self.request_auth_data()
            else:
                return False

    def start_session(self, submit_folder: Path, submit_file_regex=r'.+\.sub',
                      session_name_prefix: str = "Unknown session", session_name_suffix: str = None,
                      machines: int = 1, cores: int = 2, multithreading: bool = False,
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
                    self.remove(session)
                raise e
            else:
                raise e

    def remove(self, session: str, kill: bool = True):
        logger = logging.getLogger()
        logger.info('Running remove command for session %s', session)
        self._execute_bhpc_command(["remove", session], 'yes' if kill else 'no')

    def list(self) -> BHPCState:
        stdout, _ = self._execute_bhpc_command(['list'])
        return BHPCState.from_bhpc_message(stdout)

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

    def _execute_bhpc_command(self, arguments: List[str], command_input: str = "") -> Tuple[str, str]:
        argv = [str(self.bhpc_exe.absolute()), *arguments]
        env = os.environ.copy()
        env.update(self._get_bhpc_env())
        # env = sort_dict(env)
        logger = logging.getLogger()
        logger.debug({"action": "Starting BHPC command", "arguments": argv})
        bhpc_process = subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=self.bhpc_exe.parent, env=env,
                                        text=True, encoding="windows-1252")
        stdout, stderr = bhpc_process.communicate(input=command_input)
        if is_auth_message(stdout):
            logger.info("BHPC rejected credentials, retrying after requesting new ones")
            self._handle_auth()
            return self._execute_bhpc_command(arguments, command_input)
        else:
            return stdout, stderr

    def _handle_auth(self):
        if ((self.session_token is None or self.key is None or self.key_id is None)
                and self.request_auth_data_when_missing):
            if not self.request_auth_data():
                raise BHPCAccessError("Failed when requesting credentials after initialising without them")
        elif self.request_auth_data_when_invalid:
            if not self.request_auth_data():
                raise BHPCAccessError("Failed when requesting credentials after the old credentials became invalid"
                                      "or were never provided")
        else:
            raise BHPCAccessError("Could not obtain valid credentials for the bhpc")


def is_auth_message(response: str) -> bool:
    """Returns True if response is the BHPC error message for missing authorization
    :param response: The response from the BHPC executable
    :return: True if response is the error response, False for all other values"""
    return 'Authorization environment variables not set' in response
