from time import time
from src.logger import console, logger
from src.utils.parsing import get_concatenated_response

def process_omr(in_omr, template):
    start_time = time()

    logger.info("")
    logger.info(f" Opening image: \tResolution: {in_omr.shape}")

    template.image_instance_ops.reset_all_save_img()
    template.image_instance_ops.append_save_img(1, in_omr)

    (response_dict, final_marked, multi_marked, _, ) = template.image_instance_ops.read_omr_response(template, image=in_omr, name=None)

    omr_response = get_concatenated_response(response_dict, template)

    logger.info(f"Read Response: \n{omr_response}")
    totalTime = time()-start_time
    logger.info(f"Total time for fetching answers: {totalTime}")
    return omr_response
