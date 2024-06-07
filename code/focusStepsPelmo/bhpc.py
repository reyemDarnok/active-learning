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
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)

def start_submit_file(submit_folder: Path, session_name_prefix = "Unknown session",submit_file_regex = r".+\.sub", machines: int =1, cores: int = 2, multithreading: bool = True, notificationemail: Optional[str] = None, session_timeout: int = 6) -> str:
    logger = logging.getLogger()
    submit_folder = submit_folder.absolute()
    if not bhpc_exe.exists():
        logger.critical('BHPC is not installed. Please follow the installation instructions at https://docs.int.bayer.com/-/ensa_BHPC_docs/tutorials/#downloading-the-bhpcexe')
    with pushd(bhpc_dir):
        session = f"{session_name_prefix}{random.getrandbits(32)}"
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

def download_submission(session: str, retry_until_finished: bool = True):
    logger = logging.getLogger()

    job_finished = False
    with pushd(bhpc_dir):
        try:
            while (not job_finished) and retry_until_finished:
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
        

def is_bhpc_job_finished_status(status: str) -> bool:
    lines = status.splitlines()
    lines = lines[2:]
    for line in lines:
        if line.split()[0] != 0 or line.split()[1] != 0:
            return False
    return True