from dolfin import *
from sipTools.meshDS import meshDS

# Create a unit square mesh
nx = 21
ny = 21
p0 = Point(0.0, 0.0)
p1 = Point(1.0, 1.0)
mesh = RectangleMesh(p0, p1, nx, ny)

# create a mesh data structure with the square mesh
square_mesh = meshDS(mesh)

# print out the number of elements, nodes and edges
print "Mesh info for square mesh"
print square_mesh.getElements()
print square_mesh.getNodes()
print square_mesh.getEdges()

# plot the mesh
plot(mesh, interactive=True)
