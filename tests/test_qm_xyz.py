import os
import shlex
import shutil
import subprocess
import tempfile

from conftest import start_server


def test_qm_xyz():
    """
    Make sure that an xyz file for the QM region is written when requested.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temporary directory.
        shutil.copyfile("tests/input/adp.parm7", tmpdir + "/adp.parm7")
        shutil.copyfile("tests/input/adp.rst7", tmpdir + "/adp.rst7")
        shutil.copyfile("tests/input/emle_prod.in", tmpdir + "/emle_prod.in")

        # Copy the current environment to a new dictionary.
        env = os.environ.copy()

        # Set environment variables.
        env["EMLE_QM_XYZ_FREQUENCY"] = "2"

        # Start the server.
        server = start_server(tmpdir, env=env)

        # Create the sander command.
        command = "sander -O -i emle_prod.in -p adp.parm7 -c adp.rst7 -o emle.out"

        process = subprocess.run(
            shlex.split(command),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Make sure that the process exited successfully.
        assert process.returncode == 0

        # Make sure that an xyz file was written.
        assert os.path.isfile(tmpdir + "/qm.xyz")

        # Make sure that the file contains the expected number of frames.
        with open(tmpdir + "/qm.xyz", "r") as f:
            num_frames = 0
            for line in f:
                if line.startswith("22"):
                    num_frames += 1
        assert num_frames == 11

        # Stop the server.
        server.terminate()
