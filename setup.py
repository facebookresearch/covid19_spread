from setuptools import setup

setup(
    name="cv",
    version="0.1",
    py_modules=["cv"],
    install_requires=["Click",],
    entry_points="""
        [console_scripts]
        cv=cv:cli
    """,
)
