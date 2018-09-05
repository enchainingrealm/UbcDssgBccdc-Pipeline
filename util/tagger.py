import subprocess
import time

from py4j.java_gateway import JavaGateway

from root import from_root
from util.get_keys import get_keys


def annotate(df, observations=False, options=""):
    """
    Returns MetaMap annotations for the given DataFrame.
    :param df: the DataFrame containing the result_full_descriptions to annotate
    - required columns: {"test_key", "result_key", "obs_seq_nbr" (if
      observations is True), "result_full_description"}
    :param observations: True if the data is given at the observation level,
    False if the data is given at the test level
    :param options: MetaMap options; input string of form "-y -D" or "-yD"
    :return: a DataFrame containing the MetaMap annotations
    - columns: {"test_key", "result_key", "obs_seq_nbr" (if observations is
      True), "tags", "candidates"}
    """
    proc = subprocess.Popen(
        ["java", "-jar", from_root("libs\\MetaMapBuild.jar")],
        shell=False
    )

    # initialize variables to suppress PyCharm warnings
    gateway = None
    metamap = None
    api = None

    try:
        connected = False
        for i in range(5):
            try:
                gateway = JavaGateway()
                metamap = gateway.entry_point
                api = metamap.getApi()

                connected = True
                print(f"Connected to Java server on attempt {i + 1}")
                break
            except:
                time.sleep(1)

        if not connected:
            raise Exception("Error connecting to Java server")

        tags = []
        candidates = []

        if options.strip():
            # Process a string with additional options string
            api.setOptions(options + " -c")
        else:
            # Process a string without additional options string
            api.setOptions("-c")

        for description in df["result_full_description"]:
            result = api.processCitationsFromString(description).get(0)

            # parse result from MetaMap to tags and candidates
            tags.append(metamap.formatOneResultToString(result, "tags"))
            candidates.append(
                metamap.formatOneResultToString(result, "candidates"))

        keys = get_keys(observations)

        return_value = df.loc[:, keys]
        return_value["tags"] = tags
        return_value["candidates"] = candidates

        return return_value
    finally:
        if gateway is not None:
            gateway.shutdown()
        proc.terminate()
