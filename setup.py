from setuptools import setup, find_packages

setup(
    name="covid19_spread",
    version="0.1",
    py_modules=["covid19_spread"],
    install_requires=["Click",],
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        cv=cv:cli
        prepare-data=prepare_data:cli
        recurring=recurring:cli
    """,
)
