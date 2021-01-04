
"""

"""
const _MIXED = 0
const _LUMPED = 1
const _LAYERED = 2
const _WEAK_DISCONTINIUOS = 3
const _STRONG_DISCONTINIUOS = 4
const _FULLY_DISCONTINIUOS = 5

struct CELLSTATE
    state::Int
    state2::Int    

    function CELLSTATE(state::Int, state2::Int)
        (state == _LUMPED || state == _LAYERED) && @assert(state2==0)
        return new(state, state2)
    end

end

Base.getindex(v::Dict{Int,T}, i::CELLSTATE) where T = v[i.state2]
Base.setindex!(v::Dict{Int,T2}, value::T2, i::CELLSTATE) where {T2} = v[i.state2] = value
Broadcast.broadcastable(c::CELLSTATE) = (c,) #In order to be able to do cellstates[indeces] .= LUMPED

const LUMPED = CELLSTATE(_LUMPED, 0)
const LAYERED = CELLSTATE(_LAYERED, 0)
const FULLY_DISCONTINIUOS = CELLSTATE(_FULLY_DISCONTINIUOS, typemax(Int))

STRONG_DISCONTINIUOS(i::Int) = CELLSTATE(_STRONG_DISCONTINIUOS, i)
WEAK_DISCONTINIUOS(i::Int) = CELLSTATE(_WEAK_DISCONTINIUOS, i)
WEAK_DISCONTINIUOS_AT_INTERFACE(iint::Int) = CELLSTATE(_WEAK_DISCONTINIUOS, (Int(1)<<(iint-1)))
STRONG_DISCONTINIUOS_AT_INTERFACE(iint::Int) = CELLSTATE(_STRONG_DISCONTINIUOS, (Int(1)<<(iint-1)))

function _DISCONTINIUOS_AT_INTERFACES(iint::NTuple{N,Int}, _DISCONTINIUOS::Int) where {N}
    state = Int(0)
    for i in iint
        state |= (Int(1)<<(i-1))
    end
    return CELLSTATE(_DISCONTINIUOS, state)
end
STRONG_DISCONTINIUOS_AT_INTERFACES(iint::NTuple{N,Int}) where {N} = _DISCONTINIUOS_AT_INTERFACES(iint, _STRONG_DISCONTINIUOS)
WEAK_DISCONTINIUOS_AT_INTERFACES(iint::NTuple{N,Int}) where {N} = _DISCONTINIUOS_AT_INTERFACES(iint, _WEAK_DISCONTINIUOS)


is_fully_discontiniuos(c::CELLSTATE) = (c.state == _FULLY_DISCONTINIUOS)# && c.state2==2^ninterfaces-1)
is_weak_discontiniuos(c::CELLSTATE) = (c.state == _WEAK_DISCONTINIUOS)
is_strong_discontiniuos(c::CELLSTATE) = (c.state == _STRONG_DISCONTINIUOS)
is_discontiniuos(c::CELLSTATE) = (c.state == _WEAK_DISCONTINIUOS || c.state == _STRONG_DISCONTINIUOS || c.state == _FULLY_DISCONTINIUOS)
is_lumped(c::CELLSTATE) = (c.state == _LUMPED)
is_layered(c::CELLSTATE) = (c.state == _LAYERED)
is_mixed(c::CELLSTATE) = (c.state == _MIXED)

Base.:(>)(a::CELLSTATE, b::CELLSTATE) = a.state > b.state 
Base.:(<)(a::CELLSTATE, b::CELLSTATE) = a.state < b.state 
#Base.:(>=)(a::CELLSTATE, b::CELLSTATE) = a.state >= b.state

@inline function is_interface_active(state::CELLSTATE, iint::Int)
    is_fully_discontiniuos(state) && return true
    !is_discontiniuos(state) && return false
    
    interface_bit = ((state.state2 >> (iint-1)) & Int(1))
    return 1 == interface_bit
end

function insert_interface(state::CELLSTATE, iint::Int, ninterfaces::Int)

    new_state2 = state.state2 | (Int(1)<<(iint-1))
    
    new_state = state.state
    if new_state2 >= 2^ninterfaces-1 #all layers active
        new_state = _FULLY_DISCONTINIUOS
        new_state2 = typemax(Int)
    else
        if new_state == _LUMPED
            new_state = _WEAK_DISCONTINIUOS
        elseif new_state == _LAYERED
            new_state = _STRONG_DISCONTINIUOS
        end
    end

    return CELLSTATE(new_state, new_state2)
end

function combine_states(a::CELLSTATE, b::CELLSTATE, ninterface::Int)
    new_state = max(a.state, b.state)
    new_state2 = a.state2 | b.state2

    if new_state == _LUMPED
        new_state = _WEAK_DISCONTINIUOS
    elseif new_state == _LAYERED
        new_state = _STRONG_DISCONTINIUOS
    end

    #Check if number of active interfaces are equal to fully_discontinous state
    new_state = (new_state2 >= 2^ninterface-1) ? _FULLY_DISCONTINIUOS : new_state
    new_state2= (new_state2 >= 2^ninterface-1) ? typemax(Int) : new_state2
    return CELLSTATE(new_state, new_state2)
end

function combine_states(states::AbstractVector{CELLSTATE}, ninterfaces::Int)
    new_cell_state = states[1]
    for iint in 2:length(states)
        new_cell_state = combine_states(new_cell_state, states[iint], ninterfaces)
    end
    return new_cell_state
end

function generate_knot_vector(state::CELLSTATE, order::Int, ninterfaces::Int)
    if state.state == _LUMPED
        return generate_knot_vector(order, ninterfaces, fill(0, ninterfaces))
    elseif state.state == _LAYERED
        return generate_knot_vector(order, ninterfaces, fill(order, ninterfaces))
    elseif is_fully_discontiniuos(state)
        return generate_knot_vector(order, ninterfaces, fill(order.+1,ninterfaces))
    elseif is_weak_discontiniuos(state)
        multiplicity = generate_nmultiplicity_vector(state, ninterfaces, order)
        return generate_knot_vector(order, ninterfaces, multiplicity)
    elseif is_strong_discontiniuos(state)
        multiplicity = generate_nmultiplicity_vector(state, ninterfaces, order)
        return generate_knot_vector(order, ninterfaces, multiplicity)
    end
end

function get_active_basefunction_in_layer(ilay::Int, order::Int, state::CELLSTATE)
    if state.state == _LUMPED
        return 1:(order+1)
    elseif state.state == _LAYERED
        return (1:order+1) .+ (ilay-1)*(order)
    elseif is_fully_discontiniuos(state)
        return (1:order+1) .+ (ilay-1)*(order+1)
    elseif is_discontiniuos(state)
        addon = is_strong_discontiniuos(state) ? order : 0
        offset = 0
        for i in 1:ilay-1
            if is_interface_active(state, i) #Discontiniuos
                offset += order+1
            else #Lumped
                offset += addon
            end
        end
        (1:order+1) .+ offset
    end
end

function generate_active_layer_dofs(nlayers::Int, order::Int, dim_s::Int, states::Vector{CELLSTATE})
    active_layer_dofs = [Int[] for _ in 1:nlayers]
    dof_offset = 0
    for cp_state in states
        for ilay in 1:nlayers
            for ib in get_active_basefunction_in_layer(ilay, order, cp_state)
                for d in 1:dim_s
                    push!(active_layer_dofs[ilay], (ib-1)*dim_s + d + dof_offset)
                end
            end
        end
        
        dof_offset += ndofs_per_controlpoint(order, nlayers, nlayers-1, dim_s, cp_state)
    end
    return active_layer_dofs
end

function ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, state::CELLSTATE)
    if is_lumped(state)
        return (ooplane_order+1)*dim_s
    elseif is_layered(state)
        return (ooplane_order*nlayers + 1)*dim_s
    elseif is_fully_discontiniuos(state)
        return (ooplane_order+1)*nlayers*dim_s
    elseif is_strong_discontiniuos(state)
        dofs = 0
        for iint in 1:ninterfaces
            dofs += (ooplane_order) * dim_s
            if is_interface_active(state, iint)
                dofs += dim_s
            end
        end
        dofs += (ooplane_order+1) * dim_s
        return dofs
    elseif is_weak_discontiniuos(state)
        dofs = (ooplane_order) * dim_s
        for iint in 1:ninterfaces
            if is_interface_active(state, iint)
                dofs += dim_s
                dofs += (ooplane_order) * dim_s
            end
        end
        dofs += dim_s
        return dofs
    end
end



"""

"""
struct IGAShellData{dim_p,dim_s,T,LM<:Five.LayeredMaterial,IM<:Five.AbstractCohesiveMaterial}
    layer_materials::LM
    interface_material::IM
    viscocity_parameter::T
    orders::NTuple{dim_s,Int}
    knot_vectors::NTuple{dim_p,Vector{T}}
    thickness::T
    width::T                                
    nlayers::Int                            
    zcoords::Vector{T}                      
    initial_cellstates::Vector{CELLSTATE}
    initial_interface_damages::Matrix{T}    
    adaptable::Bool                         
    LIMIT_UPGRADE_INTERFACE::T              
    small_deformations_theory::Bool         
    nqp_inplane_order::Int
    nqp_ooplane_per_layer::Int
    nqp_interface_order::Int                  
end

function IGAShellData(;
    layer_materials::LM,
    interface_material::IM,
    orders::NTuple{dim_s,Int},
    knot_vectors::NTuple{dim_p,Vector{T}},
    thickness::T,
    initial_cellstates::Vector{CELLSTATE},
    nqp_inplane_order::Int,
    nqp_ooplane_per_layer::Int,
    viscocity_parameter::T                  = 0.0,
    width::T                                = 1.0, #Only used in 2d,
    nlayers::Int                            = nlayers(layer_materials),
    zcoords::Vector{T}                      = collect(-thickness/2:(thickness/nlayers):thickness/2),
    initial_interface_damages::Matrix{T}    = zeros(Float64, nlayers-1, length(initial_cellstates)),
    adaptable::Bool                         = false,
    LIMIT_UPGRADE_INTERFACE::T              = 0.01,
    small_deformations_theory::Bool         = false,
    nqp_interface_order::Int                = nqp_inplane_order) where {dim_p,dim_s,T,LM<:Five.LayeredMaterial,IM<:Five.AbstractCohesiveMaterial}

    #-----
    ninterfaces = nlayers-1
    ncells = length(initial_cellstates)
    @assert(size(initial_interface_damages)[1] == ninterfaces)
    @assert(size(initial_interface_damages)[2] == ncells)
    dim_s == 3 && @assert(width == 1.0)
    @show nqp_ooplane_per_layer
    return IGAShellData(layer_materials, interface_material, viscocity_parameter, orders, knot_vectors, thickness, width, nlayers, zcoords, initial_cellstates, initial_interface_damages, adaptable, LIMIT_UPGRADE_INTERFACE, small_deformations_theory, nqp_inplane_order, nqp_ooplane_per_layer, nqp_interface_order)

end

#=function IGAShellData{dim_s}(mat::LM, imat::IM, viscocity_parameter::T, orders, knot_vectors, thickness::T, width::T, nlayers::Int, cellstate::Vector{CELLSTATE}, initial_interface_damages::Matrix{T}, adaptable::Bool,small_deformations_theory::Bool, LIMIT_UPGRADE_INTERFACE::T, qp_inplane_order, nqp_ooplane_per_layer, nqp_interface) where {dim_s,LM<:AbstractMaterial,IM,T}
    ninterfaces = nlayers-1
    ncells = length(cellstate)
    @assert(size(initial_interface_damages)[1] == ninterfaces)
    @assert(size(initial_interface_damages)[2] == ncells)
    dim_s == 3 && @assert(width == 1.0)

    dim_p = dim_s-1
    zcoords = collect(-thickness/2:(thickness/nlayers):thickness/2)
    return IGAShellData{dim_p,dim_s,T,LM,IM}(
        mat, imat, viscocity_parameter,
        orders, knot_vectors,
        zcoords, thickness, width, nlayers,
        cellstate,
        initial_interface_damages, 
        adaptable, small_deformations_theory, LIMIT_UPGRADE_INTERFACE,
        qp_inplane_order, nqp_ooplane_per_layer, nqp_interface     
    )
end=#

nlayers(data::IGAShellData) = data.nlayers
JuAFEM.getncells(data::IGAShellData) = length(data.initial_cellstates)
ninterfaces(data::IGAShellData) = nlayers(data)-1
is_small_deformation_theory(data::IGAShellData) = data.small_deformations_theory

JuAFEM.nnodes_per_cell(data::IGAShellData{dim_p}) where {dim_p} = prod(data.orders[1:dim_p].+1)::Int
getmidsurface_ip(data::IGAShellData{dim_p,dim_s}) where {dim_p, dim_s} = IGA.BernsteinBasis{dim_p,data.orders[1:dim_p]}()
ooplane_order(data::IGAShellData{dim_p,dim_s}) where {dim_p,dim_s} = data.orders[dim_s]

JuAFEM.getnquadpoints(data::IGAShellData{dim_p}) where dim_p = data.nqp_inplane_order^dim_p * data.nqp_ooplane_per_layer * data.nlayers
getnquadpoints_ooplane(data::IGAShellData) = data.nqp_ooplane_per_layer * data.nlayers

getmaterial(data::IGAShellData) = return data.layer_materials
getinterfacematerial(data::IGAShellData) = return data.interface_material

getnquadpoints_inplane(data::IGAShellData{dim_p}) where {dim_p} = data.nqp_inplane_order^dim_p
getnquadpoints_ooplane_per_layer(data::IGAShellData) = data.nqp_ooplane_per_layer

getwidth(data::IGAShellData) = data.width

viscocity_parameter(data::IGAShellData) = data.viscocity_parameter
LIMIT_UPGRADE_INTERFACE(data::IGAShellData) = data.LIMIT_UPGRADE_INTERFACE