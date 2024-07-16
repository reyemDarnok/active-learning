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
from typing import Generator, Optional, Dict


class BHPC:
    def __init__(self, bhpc_exe: Path = Path("C:\\_AWS", 'actualVersion', 'bhpc.exe'), auth_data=Dict[str, str]):
        self.bhpc_exe = bhpc_exe
        assert self._verify_auth_data(auth_data)
        self.auth_data = auth_data

    @staticmethod
    def _verify_auth_data(auth_data):
        return True

    def change_auth_data(self, new_auth_data: Dict):
        self.auth_data = new_auth_data


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


def request_auth_data():
    """Prompts the user for the credentials for the BHPC and adds them to the environment if supplied"""
    logger = logging.getLogger()
    logger.debug("Asking user for credentials")
    print("To authorize against the BHPC, please copy the CLI Credentials from http://go/bhpc-prod")
    auth_data = []
    for line in stdin:
        logger.debug({"credentials_line": line})
        line = line.strip()
        if line:
            auth_data += [line]
        elif line == 'cls':
            auth_data += [line]
            break
        else:
            break
    print("Thank you for the credentials. This script can now make requests to the BHPC")
    auth_data = "\n".join(auth_data)
    setup_env_from_copy_paste(auth_data)
    logger.info("Set credentials from user input")


def setup_env_from_copy_paste(webpage_copy_paste: str):
    """Takes the credentials for the BHPC from the webpage and adds them to the environment
    :param webpage_copy_paste: The string that is placed into the clipboard by the webpage"""
    logger = logging.getLogger()
    logger.debug({"auth_data": webpage_copy_paste})
    variable_definitions = webpage_copy_paste.splitlines()[:-2]
    logger.debug({"lines": variable_definitions})
    env_vars = {vardef.split(":", 2)[1].split("=", 2)[0]: vardef.split(":", 2)[1].split("=", 2)[1].strip('"') for vardef
                in variable_definitions}
    logger.info({"new_env_vars": env_vars})
    setup_env(key_id=env_vars["AWS_ACCESS_KEY_ID"],
              key=env_vars["AWS_SECRET_ACCESS_KEY"],
              session_token=env_vars["AWS_SESSION_TOKEN"],
              proxy=env_vars["HTTPS_PROXY"],
              no_proxy=env_vars["NO_PROXY"],
              default_region=env_vars["AWS_DEFAULT_REGION"],
              ca_bundle=env_vars["AWS_DEFAULT_REGION"])


def setup_env(key_id: str, key: str, session_token: str,
              proxy: str = "http://MVHNG:jA54QWMy@10.185.190.10:8080",
              no_proxy: str = ".bayer.biz",
              default_region: str = "eu-central-1",
              ca_bundle: str = "ca-certificates.crt"):
    """Set the environment variables for the BHPC"""
    os.environ["AWS_ACCESS_KEY_ID"] = key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = key
    os.environ["AWS_SESSION_TOKEN"] = session_token
    os.environ["AWS_CA_BUNDLE"] = ca_bundle
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["NO_PROXY"] = no_proxy
    os.environ["AWS_DEFAULT_REGION"] = default_region


def start_submit_file(submit_folder: Path,
                      session_name_prefix: str = "Unknown session", session_name_suffix: Optional[str] = None,
                      submit_file_regex=r".+\.sub",
                      machines: int = 1, cores: int = 2, multithreading: bool = True,
                      notification_email: Optional[str] = None, session_timeout: int = 6) -> str:
    """Starts a session defined by submit files in the bhpc. This method assumes that the bhpc environment variables
    have already been set.
    
    :param submit_folder: The folder to search for submit files. Will be searched recursively
    :param session_name_prefix: The prefix for bhpc session names. Defaults to "Unknown session"
    :param session_name_suffix: The suffix for bhpc session names. Defaults to a random int
    :param submit_file_regex: The regex for submit file filenames. Defaults to ".+\\.sub" which is also the bhpc default
    :param machines: How many ec2 instances to use for running.
    Prefer increasing the cores count if you need more performance as that produces less overhead
    :param cores: How many cores each instance should have. Valid values are 2,4,8,16 and 96
    :param multithreading: Whether to use multithreading support in the bhpc
    :param notification_email: Who to notify when the bhpc session completes
    :param session_timeout: Maximum time for the bhpc session. Automatically enables longRun mode if over 12.
    :return: The ID of the created bhpc session"""
    logger = logging.getLogger()
    submit_folder = submit_folder.absolute()
    with pushd(bhpc_dir):
        suffix = session_name_suffix if session_name_suffix else random.getrandbits(32)
        session = f"{session_name_prefix}{suffix}"
        logger.info('Using sessionID %s', session)
        upload(submit_folder, submit_file_regex, session)
        logging.info('Running run command')
        run(session, machines, cores, multithreading, notification_email, session_timeout)
        return session


def run(session: str, machines: int = 1, cores: int = 2, multithreading: bool = False,
        notification_email: Optional[str] = None, session_timeout: int = 6):
    """Execute the run command on the BHPC
    :param machines: How many ec2 instances should the BHPC use
    :param cores: How many cores should each ec2 instance have (valid values are: 2,4,8,16,96)
    :param notification_email: Which email inbox should be notified upon completion of the BHPC Job
    :param session_timeout: When should the session time out
    :param multithreading: True if one Machine should only host one job at a time instead of one core for one job.
    Use for jobs with native multithreading
    :param session: The session id to run"""
    assert cores in (2, 4, 8, 16, 96), f"Invalid core number {cores}. Only 2,4,8,16 or 96 are permitted"
    logger = logging.getLogger()
    command_args = [str(bhpc_exe.absolute()), 'run',
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
    run_process = subprocess.run(command_args, text=True, capture_output=True)
    logger.debug(run_process.stdout)
    if is_auth_message(run_process.stdout):
        request_auth_data()
        run(session, machines, cores, multithreading, notification_email, session_timeout)
    if run_process.stdout.startswith('Session ') and run_process.stdout.endswith(' is not initialized.'):
        raise ValueError(f"Session {session} was not initialized. Start upload command for that session first")


def upload(submit_folder, submit_file_regex, session):
    """Run the upload command for the BHPC
    :param submit_folder: The parent directory for all the submit files to upload
    :param submit_file_regex: The pattern that submit files need to match to be uploaded
    :param session: The session name to use when uploading. Has to be unique on the BHPC for current jobs"""
    logger = logging.getLogger()
    logger.info('Running upload command')
    upload_process = subprocess.run([
        str(bhpc_exe.absolute()), 'upload',
        '-path', str(submit_folder),
        '-search', submit_file_regex,
        session], text=True, capture_output=True)
    logger.debug(upload_process.stdout)

    if is_auth_message(upload_process.stdout):
        request_auth_data()
        upload(submit_folder, submit_file_regex, session)


def download(session: str, wait_until_finished: bool = True, retry_interval: float = 60) -> bool:
    """Download the results of a bhpc session.
     The results will be in the directory where the submit file that started the session is.
    :param session: The sessionId of the session to download
    :param wait_until_finished: Whether to wait until the session is finished
    :param retry_interval: How long to wait between checks of the session status
    :return: False if wait_until_finished if False and the download was not ready, True otherwise"""
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
