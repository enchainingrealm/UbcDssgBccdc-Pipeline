import subprocess
import time

from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JJavaError

from root import from_root


# df - data frame with test_key, result_key, result_full_description
# options - MetaMap options. Input string of form: "-y -D" or "-yD"
from util.get_keys import get_keys


def annotate(df, observations=False, options=""):
    proc = subprocess.Popen(
        ["java", "-jar", from_root("libs\\MetaMapBuild.jar")],
        shell=False)

    try:
        attempts = 0
        successful = False

        while attempts < 5:
            try:
                gateway = JavaGateway()
                metamap = gateway.entry_point
                api = metamap.getApi()
                print(
                    "Connected to Java server on attempt " + str(attempts + 1))
                successful = True
                break
            except:
                time.sleep(1)
                attempts += 1

        if not successful:
            raise Exception("Error connecting to Java server")

        tags = []
        candidates = []

        if options.strip():
            # Process a string with additional options string
            api.setOptions(options + " -c")
        else:
            # Process a string without additional options string
            api.setOptions("-c")

        f = open(from_root("pre-tagging_log.txt"), "w")
        connected = True
        for index, row in df.iterrows():
            description = row["result_full_description"]
            test_key = row["test_key"]
            result_key = row["result_key"]
            if description is "":
                tags.append("{}")
                candidates.append("{}")
            else:
                if connected is False:
                    f.write("test_key: " + str(test_key) + " ")
                    f.write("result_key: " + str(result_key) + " ")
                    f.write("MetaMap Connection Error \n")
                    tags.append("MetaMap Connection Errors")
                    candidates.append("MetaMap Connection Errors")
                else:
                    try:
                        result = api.processCitationsFromString(
                            description).get(0)
                    except Py4JJavaError as e:
                        error_msg = e.java_exception.getMessage()
                        f.write("test_key: " + str(test_key) + " ")
                        f.write("result_key: " + str(result_key) + " ")
                        if "Index 0 out-of-bounds for length 0" in error_msg:
                            f.write("Memory Error \n")
                            tags.append("Memory Error")
                            candidates.append("Memory Error")
                            continue
                        if "Connection refused" in error_msg:
                            f.write("MetaMap Connection Error \n")
                            tags.append("MetaMap Connection Error")
                            candidates.append("MetaMap Connection Error")
                            print(
                                "Quit tagging latter rows due to MetaMap Server connection error.")
                            connected = False
                            continue
                    except Exception:
                        tags.append("Other Errors")
                        candidates.append("Other Errors")
                        f.write("test_key: " + str(test_key) + " ")
                        f.write("result_key: " + str(result_key) + " ")
                        f.write("Other Error \n")
                        continue
                    # parse result from MetaMap to tags and candidates
                    tags.append(metamap.formatOneResultToString(result, "tags"))
                    candidates.append(
                        metamap.formatOneResultToString(result, "candidates"))
        print("MetaMap server connection is: ", metamap.isConnectAPI())
        if metamap.isConnectAPI() == "true":
            api.disconnect()
            if metamap.isConnectAPI() == "false":
                print("disconnect API from MetaMap server")
        return_value = df.loc[:, ["test_key", "result_key"]]
        return_value["tags"] = tags
        return_value["candidates"] = candidates

    finally:
        gateway.shutdown()
        proc.terminate()

    return return_value
