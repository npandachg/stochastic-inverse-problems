from dolfin import *
from sipTools.meshDS import meshDS

# Read an Lshaped mesh
mesh = Mesh("tests/Lshaped.xml")

# create a mesh data structure with the Lshaped mesh
Lshaped_mesh = meshDS(mesh)

# print out the number of elements, nodes and edges
print "Mesh info for Lshaped mesh"
print Lshaped_mesh.getElements()
print Lshaped_mesh.getNodes()
print Lshaped_mesh.getEdges()

# plot the mesh
plot(mesh, interactive=True)
