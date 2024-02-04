import setuptools

description = open("README.md", "r").read()
setuptools.setup(
  name="lightnum",
  version="0.0.1",
  author="smurfd",
  author_email="smurfd@gmail.com",
  packages=["lightnum"],
  description="lite numpy",
  long_description=description,
  long_description_content_type="text/markdown",
  url="https://github.com/smurfd/lightnum",
  license='MIT',
  python_requires='>=3.8',
  extras_require={
    'testing': ["pytest",],
    'linting': ["flake8", "mypy", "pre-commit",],
  },
)
