from setuptools import setup, find_packages

setup(
    name="turn_taking",  # Replace with your project's name
    version="0.1.0",  # Replace with your version
    description="A brief description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/project",  # Replace with your project's URL
    license="MIT",  # Replace with your license if different
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


