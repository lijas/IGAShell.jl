
function generate_cohesive_oop_quadraturerule(zcoords::Vector{T}) where {T}
    
    ninterfaces = length(zcoords)-2
    ε = 1e-13

    points_top = Vector{Vec{1,T}}()
    weights_top = Vector{T}()

    points_bot = Vector{Vec{1,T}}()
    weights_bot = Vector{T}()
    #copy the oo-plane integration to each layer
    zcoords_interfaces = change_zcoord_range(zcoords)[2:end-1]
    for iinterface in 1:ninterfaces
        #Add the zcoord to the quadrature rule
        push!(points_top,  Vec( (zcoords_interfaces[iinterface] + ε, ) ))
        push!(weights_top, 1.0)

        #Add the zcoord to the quadrature rule
        push!(points_bot,  Vec( (zcoords_interfaces[iinterface] - ε, ) ))
        push!(weights_bot, 1.0)
        #push!(weights, qr.weights[qp])
    end
    return [QuadratureRule{1,RefCube,T}(weights_bot,points_bot), QuadratureRule{1,RefCube,T}(weights_top,points_top)]

end

#changes zcoord to -1 to 1
function change_zcoord_range(zcoords::Vector{T}) where T
    addon = (last(zcoords) + first(zcoords))/2
    scale = (last(zcoords) - first(zcoords))/2
    zcoords = (zcoords.-addon)/scale
    return zcoords
end


function initial_upgrade_of_dofhandler(dh::MixedDofHandler, igashell::IGAShell)

    instructions = Five.FieldDimUpgradeInstruction[]

    for (ic, cellid) in enumerate(igashell.cellset)

        cellnodes = igashell.cell_connectivity[:, ic]
        cellnode_states = adapdata(igashell).control_point_states[cellnodes]

        initial_cellnode_states = fill(LUMPED_CPSTATE, length(cellnode_states))

        if cellnode_states != initial_cellnode_states
            ndofs = ndofs_per_cell(dh, cellid)

            instr = construct_upgrade_instruction(igashell, cellid, initial_cellnode_states, cellnode_states, zeros(Float64, ndofs), zeros(Float64, ndofs))
            push!(instructions, instr)
        end

    end
    return instructions
        
end

#
generate_knot_vector(order::Int, ninterfaces::Int, nmultiplicity::Int) = generate_knot_vector(order, ninterfaces, fill(nmultiplicity, ninterfaces))

function generate_knot_vector(order::Int, ninterfaces::Int, nmultiplicity::Vector{Int})
    @assert(ninterfaces == length(nmultiplicity))

    kv = fill(-1.0, order+1)
    for i in 1:ninterfaces
        z = i/(ninterfaces+1)*2 - 1.0
        append!(kv, fill(z,nmultiplicity[i]))
    end
    append!(kv, fill(1.0,order+1))
    return kv
end

function generate_nmultiplicity_vector(state::CPSTATE, ninterfaces::Int, order::Int) 
    if is_weak_discontiniuos(state)
        return digits(state.config, base=2, pad=ninterfaces)*(order+1)
    elseif is_strong_discontiniuos(state)
        return digits(state.config, base=2, pad=ninterfaces) .+ order
    elseif is_fully_discontiniuos(state)
        return fill(order+1, ninterfaces)
    elseif is_lumped(state)
        return fill(0,ninterfaces)
    elseif is_layered(state)
        return fill(order, ninterfaces)
    else
        error("bad state")
    end
end

function generate_out_of_plane_extraction_operators(_knot_vector::Vector{T}, order, new_knots::Vector{T}, multiplicity::Vector{Int}=fill(1, length(new_knots))) where {T}
    
    @assert(length(multiplicity) == length(new_knots))
    
    knot_vector = copy(_knot_vector)
    C = Diagonal(ones(length(knot_vector)-order-1))
    for (i, new_knot) in enumerate(new_knots)
        for _ in 1:multiplicity[i]
            C1, knot_vector = IGA.knotinsertion(knot_vector, order, new_knot)
            C = C1 * C
        end
    end
    return C
end

geometryobject(ip::Interpolation, indx::Int) = collect(1:getnbasefunctions(ip))
geometryobject(ip::Interpolation{2}, indx::FaceIndex) = 1:getnbasefunctions(ip)#JuAFEM.faces(ip)[indx[2]]
geometryobject(ip::Interpolation, indx::EdgeIndex) = JuAFEM.edges(ip)[indx[2]]
geometryobject(ip::Interpolation{1}, indx::VertexIndex) = JuAFEM.vertices(ip)[indx[2]]
geometryobject(ip::Interpolation{2}, indx::EdgeIndex) = JuAFEM.faces(ip)[indx[2]]
geometryobject(ip::Interpolation{2}, indx::EdgeInterfaceIndex) = JuAFEM.faces(ip)[indx[2]]
geometryobject(ip::Interpolation{2}, indx::VertexInterfaceIndex) = JuAFEM.vertices(ip)[indx[2]]

function active_basefunctions(field_dim::Int, indx::FaceIndex)
    indx[2] == 1 && return 1
    indx[2] == 2 && return field_dim
    error("Bad index")
end

function active_basefunctions(field_dim::Int, indx::Union{VertexInterfaceIndex, EdgeInterfaceIndex})
    indx[3] == 1 && return 1
    indx[3] == 2 && return field_dim
    error("Bad index")
end

function active_basefunctions(field_dim::Int, ::EdgeIndex)
    return 1:field_dim
end

function active_basefunctions(field_dim::Int, ::VertexIndex)
    return 1:field_dim
end

function active_basefunctions(field_dim::Int, ::Int)
    return 1:field_dim
end


#igashelldofs(dh::JuAFEM.AbstractDofHandler, igashell::IGAShell, index::GeometryObject) = celldofs(dh, index[1])[igashelldofs(igashell, index)]

function igashelldofs(igashell::IGAShell{dim_p,dim_s}, index::GeometryObject, components::Vector{Int}=collect(1:dim_s)) where {dim_p,dim_s}

    cellid = index[1]
    ip = getmidsurface_ip(layerdata(igashell))

    facepoints = geometryobject(ip, index)

    face_dofs = Int[]; currentdof = 1
    for (i, nodeid) in enumerate(cellconectivity(igashell, cellid))
        nodestate = get_controlpoint_state(adapdata(igashell), nodeid)
        nnodes_per_controlpoints = ndofs_per_controlpoint(igashell, nodestate) ÷ dim_s
        if i in facepoints
            for basefunc in active_basefunctions(nnodes_per_controlpoints, index)
                for d in components
                    push!(face_dofs, currentdof + (basefunc-1)*dim_s + d -1)
                end
            end
        end
        currentdof += nnodes_per_controlpoints*dim_s
    end

    return face_dofs
end


function extrapolate(y1,y2,x1,x2,x)
    return y1 + ((x-x1)/(x2-x1))*(y2-y1)
end#



"""
Returns the vertex id below,
4__________3
|          |
|__________|
1          2
, given the vertex id 1,3,7 or 9 from the bezierelement
7____8_____9
|          |
4    5     6
|__________|
1    2     3
"""
function vertex_converter(igashell::IGAShell{2,3}, v::Int)
    if v == 1
        return 1
    elseif v == igashell.layerdata.orders[1]+1
        return 2
    elseif v == JuAFEM.nnodes_per_cell(igashell)
        return 3
    elseif v == JuAFEM.nnodes_per_cell(igashell)-igashell.layerdata.orders[1]
        return 4
    else
        error("Bad vertex")
    end
end

function vertex_converter(igashell::IGAShell{1,2}, v::Int)
    if v == 1
        return 1
    elseif v == JuAFEM.nnodes_per_cell(igashell)
        return 2
    else
        error("Bad vertex")
    end
end


"""
_convert_2_3dstate
    Takes a tensor i 2d and convertes into a stress state in 3d
    Assume plane stress
"""
function _convert_2_3dstate(σ::SymmetricTensor{2,2,T}) where T
    return SymmetricTensor{2,3,T,6}((σ[1,1], T(0.0), σ[1,2], T(0.0), T(0.0), σ[2,2]))
end