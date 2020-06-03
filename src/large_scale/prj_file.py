# create projectfile

projectfile_template = """<?xml version="1.0" encoding="utf-8" ?>
<config>
    <source_groups>
        <source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
            <name>Python Source Group</name>
            <python_environment_path>{}</python_environment_path>
            <source_extensions>
                <source_extension>.py</source_extension>
            </source_extensions>
            <source_paths>
                <source_path>{}</source_path>
            </source_paths>
            <status>enabled</status>
            <type>Python Source Group</type>
        </source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
    </source_groups>
    <version>8</version>
</config>"""

def create_projectfile(name, src_path, env_path):
    with open(f"{name}.srctrlprj", "w") as pfile:
        pfile.write(projectfile_template.format(env_path, src_path))