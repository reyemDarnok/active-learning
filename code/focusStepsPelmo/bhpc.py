import contextlib
from pathlib import Path
import os
import logging
import shutil
import random
import subprocess
import time
from subprocess import CalledProcessError
from typing import Optional

bhpc_dir = Path('C:\\_AWS', 'actualVersion')
bhpc_exe = bhpc_dir / 'bhpc.exe'
 
@contextlib.contextmanager
def pushd(new_dir):
    '''Emulates the behavior of pushd/popd. 
    During the context the current working directory will be new_dir, 
    and after closing the working directory will be restored to its old value. 
    This contextmanager can be nested and overwrites working directory changes made inside it when exiting
    
    :param new_dir The directory to move to during the context'''
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)

def start_submit_file(submit_folder: Path, 
                      session_name_prefix: str = "Unknown session", session_name_suffix: Optional[str] = None, submit_file_regex = r".+\.sub", 
                      machines: int =1, cores: int = 2, multithreading: bool = True, 
                      notificationemail: Optional[str] = None, session_timeout: int = 6) -> str:
    """Starts a session defined by submit files in the bhpc. This method assumes that the bhpc environment variables have already been set.
    
    :param submit_folder The folder to search for submit files. Will be searched recursively
    :param session_name_prefix The prefix for bhpc session names. Defaults to "Unknown session"
    :param session_name_suffix The suffix for bhpc session names. Defaults to a random int
    :param submit_file_regex The regex for submit file filenames. Defaults to ".+\.sub" which is also the bhpc default
    :param machines How many ec2 instances to use for running. Prefer increasing the cores count if you need more performance as that reduces overhead
    :param cores How many cores each instance should have. Valid values are 2,4,8,16 and 96
    :param multithreading Whether to use multithreading support in the bhpc
    :param notificationemail Who to notify when the bhpc session completes
    :param session_timeout Maximum time for the bhpc session. Automatically enables longRun mode if over 12. CURRENTLY BUGGED"""
    assert cores in (2,4,8,16,96)
    assert bhpc_exe.exists()

    logger = logging.getLogger()
    submit_folder = submit_folder.absolute()
    with pushd(bhpc_dir):
        suffix = session_name_suffix if session_name_suffix else random.getrandbits(32)
        session = f"{session_name_prefix}{suffix}"
        logger.info('Using sessionID %s', session)
        logger.info('Running upload command')
        try:
            subprocess.run([
                str(bhpc_exe.absolute()), 'upload', 
                '-path', str(submit_folder), 
                '-search', submit_file_regex, 
                session], check=True)
        except CalledProcessError as e:
            logger.error('Failed to upload bhpc job. A frequent problem is that the credentials have not been configured in the past 8 hours.')
            raise e
        
        try:
            logging.info('Running run command')
            command_args = [str(bhpc_exe.absolute()), 'run', 
                '-force',
                '-cores', str(cores), 
                '-count', str(machines),
                session]
            if multithreading:
                command_args += ['-multi']
            if notificationemail:
                command_args += ['-notificationEmail', notificationemail]
            if session_timeout > 12:
                command_args += ['-longRun']
            subprocess.run(command_args, check=True)
        except CalledProcessError as e:
            logger.error('Failed to run bhpc job. A frequent problem is that the credentials have not been configured in the past 8 hours.')
            raise e
        return session

def download_submission(session: str, wait_until_finished: bool = True)-> bool:
    """Download the results of a bhpc session The results will be in the directory where the submit file that started the session is.
    :param session The sesssionId of the session to download
    :param wait_until_finished Whether to wait until the session is finished
    :return If wait_until_finished is False return False if the session is not finished yet. Otherwise returns True after download"""
    logger = logging.getLogger()

    with pushd(bhpc_dir):
        try:
            logging.info('Running show command')
            p = subprocess.run([
                    str(bhpc_exe.absolute()), 'show', session], capture_output=True, check=True)
            if not is_bhpc_job_finished_status(p.stdout) and not wait_until_finished:
                return False
            job_finished = False
            while (not job_finished) and wait_until_finished:
                logging.info('Running show command')
                p = subprocess.run([
                    str(bhpc_exe.absolute()), 'show', session], capture_output=True, check=True)
                if is_bhpc_job_finished_status(p.stdout):
                    job_finished = True
                else:
                    time.sleep(60)
        except CalledProcessError as e:
            logger.error('Failed to check the bhpc jobs status. A frequent problem is that the credentials have not been configured in the past 8 hours.')
            raise e
        
        try:
            logging.info('Running download command')
            subprocess([
                str(bhpc_exe.absolute), 'download', session
            ])
        except CalledProcessError as e:
            logger.error('Failed to download the bhpc job. A frequent problem is that the credentials have not been configured in the past 8 hours.')
            raise e
        return True
        

def is_bhpc_job_finished_status(status: str) -> bool:
    """Checks whether the status message is a finished bhpc session
    :param status The status message as output by bhpc.exe show "sessionId"
    :return True if all jobs in the session have finished status, False otherwise"""
    lines = status.splitlines()
    lines = lines[2:]
    for line in lines:
        if line.split()[0] != 0 or line.split()[1] != 0:
            return False
    return True