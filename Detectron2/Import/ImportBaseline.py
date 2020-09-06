import numpy
import os
import pickle
import re
import sys
from numpy import array

# Example: python ImportBaseline.py model_final_997cc7
modelName = sys.argv[1]

checkpoints = os.getenv('D2_CHECKPOINTS_DIR') + '\\'

fcpp = open(os.getcwd() + '\\Baseline\\' + modelName + '.cpp', 'w')
fdataFileName = checkpoints + modelName + '.data'
fdata = open(fdataFileName, 'wb')
loaded = pickle.load(open(checkpoints + modelName + '.pkl', 'rb'))
model = loaded["model"]

fcpp.write('#include "Base.h"\n')
fcpp.write('#include <Detectron2/Import/ModelImporter.h>\n')
fcpp.write('\n')
fcpp.write('using namespace Detectron2;\n')
fcpp.write('\n')
fcpp.write('/' * 119)
fcpp.write('\n')
fcpp.write('\n')

fcpp.write('std::string ModelImporter::import_' + modelName + '() {\n')
offset = 0
for key in model:
    data = model[key]
    shape = data.shape
    numel = data.size
    data = data.reshape([numel])

    fcpp.write('\tAdd("' + key + '", ' + str(numel) + '); // ' + str(offset) + '\n')
    fdata.write(data.tobytes())
    offset += numel * 4
    assert fdata.tell() == offset, "{} != {}".format(fdata.tell(), offset)

fcpp.write('\n')
fcpp.write('\treturn DataDir() + "\\\\' + modelName + '.data";\n')
fcpp.write('}\n')
