# Modified.
"""
URDF Merge Tool - Combine two URDF robot models into a single URDF file.

This module provides functionality to merge URDF files, scale robots,
copy resource files, and update resource paths for combined models.
"""

import os
import shutil
import logging
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_XYZ = '0 0 0'
DEFAULT_CONNECTION_RPY = '0 0 0'
DEFAULT_PARENT_LINK = 'robot1_base_link'
DEFAULT_CHILD_LINK = 'robot2_base_link'

MAX_RESOURCE_SEARCH_DEPTH = 5
MAX_XML_ELEMENTS = 1000  # Safety limit

RESOURCE_DIRS = ['meshes', 'textures', 'materials']

XML_INDENT = "  "

class URDFMerger:

    def __init__(self, urdf1_path, urdf2_path):
        """
        Initialize URDF merger.
        
        Args:
            urdf1_path: Path to first URDF file
            urdf2_path: Path to second URDF file
        """
        self.urdf1_path = Path(urdf1_path)
        self.urdf2_path = Path(urdf2_path)
        
        # Create output directory
        urdf1_name = self.urdf1_path.stem
        urdf2_name = self.urdf2_path.stem
        current_file_path = Path(__file__).resolve().parent
        root_path = current_file_path.parent.parent
        asset_path = root_path / 'src' / 'assets'
        
        self.output_dir = asset_path / 'urdf' / f"{urdf1_name}_combine_{urdf2_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.meshes_dir = self.textures_dir = self.output_dir
        self.meshes_dir.mkdir(exist_ok=True)
        self.textures_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")


    def _scale_robot(self, root, prefix_name, scale_factor=5.0):
        """
        Scale robot elements by a given factor.
        
        Args:
            root: XML root element
            prefix_name: Prefix to identify robot elements
            scale_factor: Scaling factor
        """
        logger.info(f"Scaling robot with prefix '{prefix_name}' by factor {scale_factor}")
          
        scale_factor_3 = scale_factor ** 3  # Volume scaling
        scale_factor_5 = scale_factor ** 5  # Inertia scaling
        
        robot_links = []
        robot_joints = []
        
        for elem in root:
            if elem.tag == 'link' and elem.get('name', '').startswith(prefix_name):
                robot_links.append(elem)
            elif elem.tag == 'joint' and elem.get('name', '').startswith(prefix_name):
                robot_joints.append(elem)
        
        logger.info(f"Found {len(robot_links)} links and {len(robot_joints)} joints to scale")

        mesh_count = self._scale_mesh_geometries(robot_links, scale_factor)
        logger.info(f"Scaled {mesh_count} mesh geometries")
        
        inertial_count = self._scale_inertial_parameters(robot_links, scale_factor, scale_factor_3, scale_factor_5)
        logger.info(f"Scaled {inertial_count} inertial parameters")
        
        joint_count = self._scale_joint_positions(robot_joints, scale_factor)
        logger.info(f"Scaled {joint_count} joint positions")


    def _scale_mesh_geometries(self, links, scale_factor):
        """Scale mesh geometries in visual and collision elements."""
        mesh_count = 0
        
        for link in links:
            # Scale visual geometry
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh.set('scale', f"{scale_factor} {scale_factor} {scale_factor}")
                        mesh_count += 1

            # Scale collision geometry
            collision = link.find('collision')
            if collision is not None:
                geometry = collision.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh.set('scale', f"{scale_factor} {scale_factor} {scale_factor}")
                        mesh_count += 1
        
        return mesh_count
    

    def _scale_inertial_parameters(self, links, scale_factor, scale_factor_3, scale_factor_5):
        """Scale inertial parameters including mass and inertia."""
        inertial_count = 0

        for link in links:
            inertial = link.find('inertial')
            if inertial is not None:
                # Scale mass
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    try:
                        original_mass = float(mass_elem.get('value', 0))
                        new_mass = original_mass * scale_factor_3
                        mass_elem.set('value', str(new_mass))
                    except ValueError:
                        logger.warning(f"Invalid mass value in {link.get('name')}")

                # Scale origin
                origin_elem = inertial.find('origin')
                if origin_elem is not None and origin_elem.get('xyz'):
                    try:
                        xyz_parts = origin_elem.get('xyz').split()
                        if len(xyz_parts) == 3:
                            new_xyz = [str(float(coord) * scale_factor) for coord in xyz_parts]
                            origin_elem.set('xyz', ' '.join(new_xyz))
                    except ValueError:
                        logger.warning(f"Invalid origin coordinates in link: {link.get('name', 'unknown')}")
                    
                # Scale inertia
                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                        value = inertia_elem.get(attr)
                        if value and value != '0':
                            try:
                                new_value = float(value) * scale_factor_5
                                inertia_elem.set(attr, str(new_value))
                            except ValueError:
                                logger.warning(f"Invalid inertia value {attr}={value} in {link.get('name')}")
            
                inertial_count += 1

        return inertial_count
        

    def _scale_joint_positions(self, joints, scale_factor):
        """Scale joint origin positions."""
        joint_count = 0
        
        for joint in joints:
            origin_elem = joint.find('origin')
            if origin_elem is not None and origin_elem.get('xyz'):
                try:
                    xyz_parts = origin_elem.get('xyz').split()
                    if len(xyz_parts) == 3:
                        new_xyz = [str(float(coord) * scale_factor) for coord in xyz_parts]
                        origin_elem.set('xyz', ' '.join(new_xyz))
                        joint_count += 1
                except ValueError:
                    logger.warning(f"Invalid joint origin coordinates in joint: {joint.get('name', 'unknown')}")
        
        logger.info(f"Scaled {joint_count} joint positions")
        return joint_count
    

    def _find_resource_directory(self, urdf_path, resource_type='meshes', max_depth=MAX_RESOURCE_SEARCH_DEPTH):
        """
        Find resource directory by searching up the directory tree.
        
        Args:
            urdf_path: Path to URDF file
            resource_type: Type of resource to find
            max_depth: Maximum depth to search
            
        Returns:
            Path to resource directory or None if not found
        """
        current_dir = urdf_path.parent

        for _ in range(max_depth):
            resource_dir = current_dir / resource_type
            if resource_dir.exists() and resource_dir.is_dir():
                return current_dir
            
            parent_dir = current_dir.parent
            if parent_dir == current_dir: # Reached root
                break
            current_dir = parent_dir
            
        return None


    def _extract_resource_path(self, resource_url):
        """
        Extract relative resource path from package:// URL.
        
        Args:
            resource_url: Resource URL starting with package://
            
        Returns:
            Relative resource path
        """
        if resource_url.startswith('package://'):
            resource_url = resource_url.replace('package://', '')

        path_parts = resource_url.split('/')
        
        for i, part in enumerate(path_parts):
            if part in ['meshes', 'textures']:
                return '/'.join(path_parts[i:])
            
        return ""


    def _collect_used_resources(self, root):
        """
        Collect all resources used in the URDF.
        
        Args:
            root: XML root element
            
        Returns:
            Set of (resource_type, relative_path) tuples
        """
        resources_to_copy = set()

        # Find mesh resources
        for mesh in root.findall('.//mesh'):
            filename = mesh.get('filename', '')
            if filename.startswith('package://'):
                relative_path = self._extract_resource_path(filename)
                resources_to_copy.add(('mesh', relative_path))

        # Find texture resources
        for texture in root.findall('.//texture'):
            filename = texture.get('filename', '')
            if filename.startswith('package://'):
                relative_path = self._extract_resource_path(filename)
                resources_to_copy.add(('texture', relative_path))

        # Find material textures
        for material in root.findall('.//material'):
            texture = material.find('texture')
            if texture is not None:
                filename = texture.get('filename', '')
                if filename.startswith('package://'):
                    relative_path = self._extract_resource_path(filename)
                    resources_to_copy.add(('texture', relative_path))
        logger.info(f"Found {len(resources_to_copy)} resources to copy")
        return resources_to_copy


    def _copy_used_resources(self, root, urdf1_path, urdf2_path):
        """
        Copy all resources used in the combined URDF.
        
        Args:
            root: XML root element
            urdf1_path: Path to first URDF
            urdf2_path: Path to second URDF
            
        Returns:
            List of copied files as (resource_type, relative_path, target_path)
        """
        logger.info("Copying used resources")
        
        resources_to_copy = self._collect_used_resources(root)
        
        urdf1_root = self._find_resource_directory(urdf1_path)
        urdf2_root = self._find_resource_directory(urdf2_path)

        logger.info(f"URDF1 root: {urdf1_root}")
        logger.info(f"URDF2 root: {urdf2_root}")

        copied_files = []
        
        for resource_type, relative_path in resources_to_copy:
            source_path = None
            source_root = None

            # Try to find in URDF1 resources
            if urdf1_root:
                potential_path = urdf1_root / relative_path
                if potential_path.exists():
                    source_path = potential_path
                    source_root = urdf1_root
            
            # Try to find in URDF2 resources
            if source_path is None and urdf2_root:
                potential_path = urdf2_root / relative_path
                if potential_path.exists():
                    source_path = potential_path
                    source_root = urdf2_root
            
            if source_root and source_path:
                if resource_type == 'mesh':
                    target_dir = self.meshes_dir
                else:  # texture
                    target_dir = self.textures_dir
                
                # Create subdirectories if needed
                relative_dir = os.path.dirname(relative_path)
                if relative_dir:
                    target_subdir = target_dir / relative_dir
                    target_subdir.mkdir(parents=True, exist_ok=True)
                
                target_path = target_dir / relative_path
                try:
                    shutil.copy2(source_path, target_path)
                    copied_files.append((resource_type, relative_path, str(target_path)))
                    logger.info(f"Copied {resource_type}: {relative_path}")
                except Exception as e:
                    logger.error(f"Failed to copy {source_path}: {e}")
        
        logger.info(f"Successfully copied {len(copied_files)} files")
        return copied_files


    def _update_resource_paths(self, root):
        """
        Update resource paths in the URDF to use relative paths.
        
        Args:
            root: XML root element
        """
        logger.info("Updating resource paths")

        # Update mesh paths
        for mesh in root.findall('.//mesh'):
            filename = mesh.get('filename', '')
            if filename.startswith('package://'):
                relative_path = self._extract_resource_path(filename)
                new_path = f"./{relative_path}"
                mesh.set('filename', new_path)
                logger.debug(f"Updated mesh: {filename} -> {new_path}")

        # Update texture paths
        for texture in root.findall('.//texture'):
            filename = texture.get('filename', '')
            if filename.startswith('package://'):
                relative_path = self._extract_resource_path(filename)
                new_path = f"./{relative_path}"
                texture.set('filename', new_path)
                logger.debug(f"Updated texture: {filename} -> {new_path}")

        # Update material texture paths
        for material in root.findall('.//material'):
            texture = material.find('texture')
            if texture is not None:
                filename = texture.get('filename', '')
                if filename.startswith('package://'):
                    relative_path = self._extract_resource_path(filename)
                    new_path = f"./{relative_path}"
                    texture.set('filename', new_path)
                    logger.debug(f"Updated material texture: {filename} -> {new_path}")


    def _clean_xml_format(self, xml_str):
        """
        Clean XML formatting by removing empty lines and trailing whitespace.
        
        Args:
            xml_str: XML string to clean
            
        Returns:
            Cleaned XML string
        """
        lines = []
        for line in xml_str.split('\n'):
            stripped_line = line.strip()
            if stripped_line:
                lines.append(line.rstrip())
        return '\n'.join(lines)


    def _prefix_robot_elements(self, elements, prefix):
        """
        Add prefix to robot elements to avoid naming conflicts.
        
        Args:
            elements: List of XML elements
            prefix: Prefix to add
        """
        for elem in elements:
            old_name = elem.get('name', '')
            if not old_name.startswith(prefix):
                new_name = f"{prefix}_{old_name}"
                elem.set('name', new_name)
            
            # Update joint references
            if elem.tag == 'joint':
                self._update_joint_references(elem, prefix)


    def _update_joint_references(self, joint, prefix):
        """
        Update joint parent/child references.
        
        Args:
            joint: Joint element
            prefix: Prefix to add to link names
        """
        parent = joint.find('parent')
        if parent is not None:
            link_name = parent.get('link')
            if not link_name.startswith(prefix):
                parent.set('link', f"{prefix}_{link_name}")
        
        child = joint.find('child')
        if child is not None:
            link_name = child.get('link')
            if not link_name.startswith(prefix):
                child.set('link', f"{prefix}_{link_name}")
        
        # Handle mimic joints
        mimic = joint.find('mimic')
        if mimic is not None:
            joint_name = mimic.get('joint')
            if joint_name and not joint_name.startswith(prefix):
                mimic.set('joint', f"{prefix}_{joint_name}")


    def merge_urdfs(
        self, 
        urdf1_path, 
        urdf2_path, 
        urdf1_name='robot1', 
        urdf2_name='robot2', 
        output_name='combined_robot',
        parent_link=DEFAULT_PARENT_LINK,
        child_link=DEFAULT_CHILD_LINK,
        connection_xyz=DEFAULT_CONNECTION_XYZ, 
        connection_rpy=DEFAULT_CONNECTION_RPY, 
        scale_robot=None
    ):
        """
        Merge two URDF files into a single URDF.
        
        Args:
            urdf1_path: Path to first URDF file
            urdf2_path: Path to second URDF file
            urdf1_name: Name prefix for first robot
            urdf2_name: Name prefix for second robot
            output_name: Name of output URDF
            parent_link: Parent link for connection joint
            child_link: Child link for connection joint
            connection_xyz: XYZ coordinates for connection joint
            connection_rpy: RPY rotation for connection joint
            scale_robot: Scaling factor for second robot (if None, no scaling)
            
        Returns:
            Path to merged URDF file or None if failed
        """
        try:
            logger.info(f"Starting URDF merge: {urdf1_name} + {urdf2_name}")
            logger.info(f"Loading URDF files: {urdf1_path}, {urdf2_path}")

            tree1 = ET.parse(urdf1_path)
            tree2 = ET.parse(urdf2_path)
        
        except Exception as e:
            logger.error(f"Error parsing URDF files: {e}")
            return None

        root = ET.Element('robot', {'name': output_name})

        # Add robot1 elements with prefix
        logger.info(f"Adding {urdf1_name} elements")
        robot1_root = tree1.getroot()
        robot1_elements = [elem for elem in robot1_root if elem.tag in ['link', 'joint']]
        self._prefix_robot_elements(robot1_elements, urdf1_name)

        for elem in robot1_elements:
            root.append(elem)

        # Add robot2 elements with prefix
        logger.info(f"Adding {urdf2_name} elements")
        robot2_root = tree2.getroot()
        robot2_elements = [elem for elem in robot2_root if elem.tag in ['link', 'joint']]
        self._prefix_robot_elements(robot2_elements, urdf2_name)

        for elem in robot2_elements:
            root.append(elem)
        
        # Scale robot if requested
        if scale_robot is not None:
            logger.info(f"Scaling {urdf2_name} by factor {scale_robot}")
            self._scale_robot(root, urdf2_name, scale_robot)

        # Create connection joint
        logger.info(f"Creating connection joint between {parent_link} and {child_link}")
        connection_joint_name = f"{urdf1_name}_{urdf2_name}_connection_joint"
        
        parent_link_final = f"{urdf1_name}_{parent_link}" if not parent_link.startswith(urdf1_name) else parent_link
        child_link_final = f"{urdf2_name}_{child_link}" if not child_link.startswith(urdf2_name) else child_link

        connection_joint = ET.SubElement(root, 'joint', {
            'name': connection_joint_name, 
            'type': 'fixed'
        })
        ET.SubElement(connection_joint, 'parent', {
            'link': parent_link_final
        })
        ET.SubElement(connection_joint, 'child', {
            'link': child_link_final
        })
        ET.SubElement(connection_joint, 'origin', {
            'xyz': connection_xyz,
            'rpy': connection_rpy
        })

        # Copy and update resources
        copied_files = self._copy_used_resources(root, Path(urdf1_path), Path(urdf2_path))
        self._update_resource_paths(root)

        # Generate XML string
        rough_string = ET.tostring(root, encoding='utf-8')
        xml_str = minidom.parseString(rough_string).toprettyxml(indent=XML_INDENT)
        xml_str = self._clean_xml_format(xml_str)

        # Save merged URDF
        file_path = self.output_dir / f'{output_name}.urdf'
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            logger.info(f"URDF merged and saved: {file_path}")
            logger.info(f"Connection joint: {connection_joint_name}")
            logger.info(f"Parent link: {parent_link_final}")
            logger.info(f"Child link: {child_link_final}")
            logger.info(f"Connection position: {connection_xyz}")
            logger.info(f"Connection rotation: {connection_rpy}")
            logger.info(f"Copied {len(copied_files)} resource files")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save merged URDF: {e}")
            return None


if __name__ == "__main__":
    # Define paths
    current_file_path = Path(__file__).resolve().parent
    root_path = current_file_path.parent.parent
    asset_path = root_path / 'src' / 'assets'

    urdf1_path = asset_path / 'urdf' / '卫星1号urdf.SLDASM' / 'urdf' / '卫星1号urdf.SLDASM.urdf'
    urdf2_path = asset_path / 'urdf' / 'arms' / 'starlink_sldasm9' / 'arm.urdf'

    # Create merger instance
    urdf_merger = URDFMerger(urdf1_path, urdf2_path)
    
    # Merge URDFs
    merged_urdf_path = urdf_merger.merge_urdfs(
        urdf1_path, 
        urdf2_path, 
        urdf1_name = os.path.splitext(os.path.basename(urdf1_path))[0], 
        urdf2_name = os.path.splitext(os.path.basename(urdf2_path))[0], 
        output_name = f"{os.path.splitext(os.path.basename(urdf1_path))[0]}_combine_{os.path.splitext(os.path.basename(urdf2_path))[0]}",
        parent_link = 'base_link',
        child_link = 'starlink_space_manipulator_base_link',
        connection_xyz = '0 0 0',
        connection_rpy = '0 0 1.574',
        scale_robot = 0.6
    )

    if merged_urdf_path:
        logger.info(f"Merge completed successfully: {merged_urdf_path}")
    else:
        logger.error("Merge failed")
