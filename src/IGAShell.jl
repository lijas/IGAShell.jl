module IgAShell

export VertexInterfaceIndex, EdgeInterfaceIndex

using Five
using IGA
using StaticArrays
using TimerOutputs

"""
EdgeInterfaceIndex
    Specifiying interface on cell.
    Used when for example applying boundary conditions


    ----+ interface = 2
        ?
    ----+ interface = 3 ??
        ?
    ----+ interface = 1
"""

struct EdgeInterfaceIndex 
    idx::Tuple{Int,Int,Int} # cellid, edgeid, interfaceid
end
EdgeInterfaceIndex(a::Int, b::Int, c::Int) = EdgeInterfaceIndex((a,b,c))
EdgeInterfaceIndex(array_or_set, c::Int) = [EdgeInterfaceIndex(a,b,c) for (a,b) in array_or_set]
Base.getindex(I::EdgeInterfaceIndex, i::Int) = I.idx[i]
Base.iterate(I::EdgeInterfaceIndex, state::Int=1) = (state==4) ?  nothing : (I[state], state+1)
Base.in(v::Tuple{Int, Int, Int}, s::Set{EdgeInterfaceIndex}) = in(EdgeInterfaceIndex(v), s)

struct VertexInterfaceIndex 
    idx::Tuple{Int,Int,Int} # cellid, edgeid, interfaceid
end
VertexInterfaceIndex(a::Int, b::Int, c::Int) = VertexInterfaceIndex((a,b,c))
VertexInterfaceIndex(array_or_set, c::Int) = [VertexInterfaceIndex(a,b,c) for (a,b) in array_or_set]
Base.getindex(I::VertexInterfaceIndex, i::Int) = I.idx[i]
Base.iterate(I::VertexInterfaceIndex, state::Int=1) = (state==4) ?  nothing : (I[state], state+1)
Base.in(v::Tuple{Int, Int, Int}, s::Set{VertexInterfaceIndex}) = in(VertexInterfaceIndex(v), s)

"""
GeometryObjectVectors
    Used when non knowing the type of "INDEX" that will be used (faceIndex, EdgeIndex etc.)
"""
const GeometryObjectVectors = Union{Array{Int,1}, Array{EdgeIndex,1}, Array{FaceIndex,1}, Array{VertexIndex,1}, Array{EdgeInterfaceIndex,1}, Array{VertexInterfaceIndex,1}}
const GeometryObject = Union{Int, EdgeIndex, FaceIndex, VertexIndex, EdgeInterfaceIndex, VertexInterfaceIndex}

include("higherorderlagrange.jl") #TODO: Move to sperate package instead

include("igashell_values.jl") 
include("igashell_data.jl")

"""
Handles all the ploting and exporting of the IGAShell to  VTK 
"""
struct IGAShellVTK{dim_p,dim_s,T,ISV<:IGAShellValues}

    # Stores all the shape values for the underlying FE-cell
    # that is used visualize the IGA-cell.
    cell_values_plot::ISV

    n_plot_points_dim::NTuple{dim_s,Int}

    #Cache of vtkgrid
    node_coords::Vector{Vec{dim_s,T}}
    cls::Vector{MeshCell{VTKCellType,Vector{Int}}}
end

include("igashell_adaptivity.jl")
include("igashell_sr.jl")
include("igashell_integrationdata.jl")
include("igashell_main.jl")
include("igashell_vtk.jl")
include("igashell_upgrade.jl")
include("igashell_external_force.jl")
#include("igashell_material.jl")
include("igashell_weakbc.jl")
include("igashell_utils.jl")
include("igashell_autodiff.jl")

end