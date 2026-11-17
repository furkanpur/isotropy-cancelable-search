"""Import hygiene: numpy-only attack modules must not pull in torch."""
import subprocess
import sys


def test_attacks_package_does_not_import_torch():
    code = (
        "import sys\n"
        "import ijiss_btp.attacks\n"
        "assert 'torch' not in sys.modules, 'torch was imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_attacks_lazy_names_still_resolve():
    from ijiss_btp.attacks import coordinate_wise_inversion_attack, invert_linear_template

    assert callable(invert_linear_template)
    assert callable(coordinate_wise_inversion_attack)
