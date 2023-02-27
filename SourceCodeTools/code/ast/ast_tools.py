import ast
import logging

import astunparse

from SourceCodeTools.code.annotator_utils import to_offsets, resolve_self_collision, adjust_offsets2


def test_get_mentions():
    test_cases = [
        'def pretty(xmls) :\n    try:\n        tree = ElementTree.XML(xmls)\n        logger.debug("Success to parsing xml")\n        return minidom.parseString(ElementTree.tostring(tree)).toprettyxml(indent=\'    \')\n    except ElementTree.ParseError:\n        logger.warning("Fail to parsing xml")\n        return f"""Include invalid token\n\n        {xmls}\n        """\n',
        'def addToilService(self, config, role, keyPath , preemptable ):\n\n\n        # If keys are rsynced, then the mesos-agent needs to be started after the keys have been\n        # transferred. The waitForKey.sh script loops on the new VM until it finds the keyPath file, then it starts the\n        # mesos-agent. If there are multiple keys to be transferred, then the last one to be transferred must be\n        # set to keyPath.\n        MESOS_LOG_DIR = \'--log_dir=/var/lib/mesos \'\n        LEADER_DOCKER_ARGS = \'--registry=in_memory --cluster={name}\'\n        # --no-systemd_enable_support is necessary in Ubuntu 16.04 (otherwise,\n        # Mesos attempts to contact systemd but can\'t find its run file)\n        WORKER_DOCKER_ARGS = \'--work_dir=/var/lib/mesos --master={ip}:5050 --attributes=preemptable:{preemptable} --no-hostname_lookup --no-systemd_enable_support\'\n\n        if self.clusterType == \'mesos\':\n            if role == \'leader\':\n                entryPoint = \'mesos-master\'\n                entryPointArgs = MESOS_LOG_DIR + LEADER_DOCKER_ARGS.format(name=self.clusterName)\n            elif role == \'worker\':\n                entryPoint = \'mesos-agent\'\n                entryPointArgs = MESOS_LOG_DIR + WORKER_DOCKER_ARGS.format(ip=self._leaderPrivateIP,\n                                                            preemptable=preemptable)\n            else:\n                raise RuntimeError("Unknown role %s" % role)\n        elif self.clusterType == \'kubernetes\':\n            if role == \'leader\':\n                # We need *an* entry point or the leader container will finish\n                # and go away, and thus not be available to take user logins.\n                entryPoint = \'sleep\'\n                entryPointArgs = \'infinity\'\n            else:\n                raise RuntimeError(\'Toil service not needed for %s nodes in a %s cluster\',\n                    role, self.clusterType)\n        else:\n            raise RuntimeError(\'Toil service not needed in a %s cluster\', self.clusterType)\n\n        if keyPath:\n            entryPointArgs = keyPath + \' \' + entryPointArgs\n            entryPoint = "waitForKey.sh"\n        customDockerInitCommand = customDockerInitCmd()\n        if customDockerInitCommand:\n            entryPointArgs = " ".join(["\'" + customDockerInitCommand + "\'", entryPoint, entryPointArgs])\n            entryPoint = "customDockerInit.sh"\n\n        config.addUnit(f"toil-{role}.service", content=textwrap.dedent(f\'\'\'\\\n            [Unit]\n            Description=toil-{role} container\n            After=docker.service\n            After=create-kubernetes-cluster.service\n\n            [Service]\n            Restart=on-failure\n            RestartSec=2\n            ExecStartPre=-/usr/bin/docker rm toil_{role}\n            ExecStartPre=-/usr/bin/bash -c \'{customInitCmd()}\'\n            ExecStart=/usr/bin/docker run \\\\\n                --entrypoint={entryPoint} \\\\\n                --net=host \\\\\n                -v /var/run/docker.sock:/var/run/docker.sock \\\\\n                -v /var/lib/mesos:/var/lib/mesos \\\\\n                -v /var/lib/docker:/var/lib/docker \\\\\n                -v /var/lib/toil:/var/lib/toil \\\\\n                -v /var/lib/cwl:/var/lib/cwl \\\\\n                -v /tmp:/tmp \\\\\n                -v /opt:/opt \\\\\n                -v /etc/kubernetes:/etc/kubernetes \\\\\n                -v /etc/kubernetes/admin.conf:/root/.kube/config \\\\\n                --name=toil_{role} \\\\\n                {applianceSelf()} \\\\\n                {entryPointArgs}\n            \'\'\'))\n',
        'def printHelp(modules) :\n    name = os.path.basename(sys.argv[0])\n    descriptions = \'\\n        \'.join(f\'{cmd} - {get_or_die(mod, "__doc__").strip()}\' for cmd, mod in modules.items() if mod)\n    print(textwrap.dedent(f"""\n        Usage: {name} COMMAND ...\n               {name} --help\n               {name} COMMAND --help\n\n        Where COMMAND is one of the following:\n\n        {descriptions}\n        """[1:]))\n'
    ]


def get_mentions(function, root, mention):
    """
    Find all mentions of a variable in the function's body
    :param function: string that contains function's body
    :param root: body parsed with ast package
    :param mention: the name of a variable to look for
    :return: list of offsets where the variable is mentioned
    """
    mentions = []

    for node in ast.walk(root):
        if isinstance(node, ast.Name): # a variable or a ...
            if node.id == mention:
                offset = to_offsets(
                    function, [
                        (node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "mention")
                    ],
                    as_bytes=True,
                )

                if len(offset) > 0:
                    if function[offset[-1][0]: offset[-1][1]] == node.id:
                        mentions.extend(offset)
                    else:
                        # TODO
                        # disable variable search inside fstrings to avoid these errors
                        # logging.warning("Skipping offset, does not align with the source code")
                        pass

    # hack for deduplication
    # the origin of duplicates is still unknown
    # it apears that mention contain false alarms....
    mentions = resolve_self_collision(mentions)

    return mentions


def get_descendants(function, children):
    """

    :param function: function string
    :param children: List of targets.
    :return: Offsets for attributes or names that are used as target for assignment operation. Subscript, Tuple and List
    targets are skipped.
    """
    descendants = []

    # if isinstance(children, ast.Tuple):
    #     descendants.extend(get_descendants(function, children.elts))
    # else:
    for chld in children:
        # for node in ast.walk(chld):
        node = chld
        if isinstance(node, ast.Attribute) or isinstance(node, ast.Name):
        # if isinstance(node, ast.Name):
            offset = to_offsets(function,
                                [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "new_var")], as_bytes=True)
            # descendants.append((node.id, offset[-1]))
            if isinstance(node, ast.Attribute):
                actual_name = astunparse.unparse(node).strip()
            else:
                actual_name = node.id
            if function[offset[-1][0]:offset[-1][1]] == actual_name:
                descendants.append((function[offset[-1][0]:offset[-1][1]], offset[-1]))
            else:
                # logging.warning("Skipping offset, does not align with the source code")
                pass
        # elif isinstance(node, ast.Tuple):
        #     descendants.extend(get_descendants(function, node.elts))
        elif isinstance(node, ast.Subscript) or isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            pass # skip for now
        else:
            raise Exception("")

    return descendants



def get_declarations(function_):
    """

    :param function:
    :return:
    """
    function = function_.lstrip()
    initial_strip = function_[:len(function_) - len(function)]

    root = ast.parse(function)

    declarations = {}
    added = set()

    for node in ast.walk(root):
        if isinstance(node, ast.arg): # function argument
            # TODO
            # not quite sure why this if statement was needed, but there should be no annotations in the code
            if node.annotation is None:
                offset = to_offsets(
                    function, [
                        (node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "arg")
                    ],
                    as_bytes=True
                )

                assert function[offset[-1][0]:offset[-1][1]] == node.arg, f"{function[offset[-1][0]:offset[-1][1]]} != {node.arg}"

                declarations[offset[-1]] = get_mentions(function, root, node.arg)
                added.add(node.arg) # mark variable name as seen
        elif isinstance(node, ast.Assign):
            desc = get_descendants(function, node.targets)

            for d in desc:
                if d[0] not in added:
                    mentions = get_mentions(function, root, d[0])
                    valid_mentions = list(filter(lambda mention: mention[0] >= d[1][0], mentions))
                    declarations[d[1]] = valid_mentions
                    added.add(d[0])

    initial_strip_len = len(initial_strip)
    declarations = {
        adjust_offsets2([key], initial_strip_len)[0]: adjust_offsets2(val, initial_strip_len) for key, val in declarations.items()
    }

    return declarations




if __name__ == "__main__":
    f = """def get_signature_declarations(function):
    root = ast.parse(function)

    declarations = {}
    
    function = 4

    for node in ast.walk(root):
        if isinstance(node, ast.arg):
            if node.annotation is None:
                offset = to_offsets(function, [(node.lineno, node.end_lineno, node.col_offset, node.end_col_offset, "arg")])
                # print(ast.dump(node), node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
                declarations[offset[-1]] = get_mentions(function, root, node.arg)
                print(declarations)
    """
    declarations = get_declarations(f)

    for dec, mentions in declarations.items():
        print(f"{f[dec[0]: dec[1]]} {dec}", end=": ")
        for m in mentions:
            print(f"{f[m[0]: m[1]]} {m}", end="\t ")
        print()

