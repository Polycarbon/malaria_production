import time,logging

from app2 import get_respone,log
from model.load import init
from src.DetectorThread import Detector, PROPER_REGION, RESNET
from src.Management import Management
from src.ObjectMapperThread import ObjectMapper
from src.PreprocessorThread import Preprocessor

# log = logging.getLogger("main")

if __name__ == "__main__":

    model, graph = None,None
    manager = Management()
    manager.init("videos/manual_1-movie-resize.mp4")
    detector = Detector(
        manager=manager, mode=PROPER_REGION, model=model, graph=graph
    )
    ppc_worker = Preprocessor(manager, detector)
    map_worker = ObjectMapper(manager)

    ppc_worker.start()
    map_worker.start()

    # ppc_worker.join()
    print("ppc_worker end")
    # map_worker.join()
    while 1:
        isFinish = manager.get_finish()
        if isFinish:
            manager.saveFile()
            res = get_respone(manager.get_result())
            log.info("data respone: {} - head(5):{}".format(len(res["data"]), res["data"][:5]))
            manager.cap_release()
            break

print("END")
