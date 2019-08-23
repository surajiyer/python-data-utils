PYTHON_LIBS_PATH=$1
if [ -z "${PYTHON_LIBS_PATH}" ] || [ ! -d "${PYTHON_LIBS_PATH}" ]
then
    mkdir ./pythonlibs
    cd ./pythonlibs
    PYTHON_LIBS_PATH="."
    git clone https://gitlab.office.intern/journey-chain-analytics/python-data-utils.git
fi
cd "${PYTHON_LIBS_PATH}"
docker build -t datascience -f "./python-data-utils/Dockerfile" .