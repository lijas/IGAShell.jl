export IGAShellData

"""

"""
const _MIXED = 0
const _LUMPED = 1
const _LAYERED = 2
const _WEAK_DISCONTINIUOS = 3
const _STRONG_DISCONTINIUOS = 4
const _FULLY_DISCONTINIUOS = 5

struct CPSTATE
    state::Int
    config::Int    

    function CPSTATE(state::Int, config::Int)
        @assert(state != _MIXED)
        if config == 0
            @assert(state == _LUMPED || state == _LAYERED)
        else
            @assert(state != _LUMPED && state != _LAYERED)
        end 
        return new(state, config)
    end

end

struct CELLSTATE
    state::Int
    cpstates::Vector{CPSTATE}
end

get_cpstate(state::CELLSTATE, i::Int) = is_mixed(state) ? state.cpstates[i] : first(state.cpstates)

CELL_OR_CPSTATE = Union{CELLSTATE, CPSTATE}

Base.getindex(v::Dict{Int,T}, i::CELLSTATE) where T = v[i.config]
Base.setindex!(v::Dict{Int,T2}, value::T2, i::CELLSTATE) where {T2} = v[i.config] = value
Broadcast.broadcastable(c::CELLSTATE) = (c,) #In order to be able to do cellstates[indeces] .= LUMPED
Broadcast.broadcastable(c::CPSTATE) = (c,) #In order to be able to do cellstates[indeces] .= LUMPED

const LUMPED_CPSTATE = CPSTATE(_LUMPED, 0)
const LAYERED_CPSTATE = CPSTATE(_LAYERED, 0)
const FULLY_DISCONTINIUOS_CPSTATE = CPSTATE(_FULLY_DISCONTINIUOS, typemax(Int))
STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(iint::Int)  = _DISCONTINIUOS_AT_INTERFACES_CPSTATE((iint,), _STRONG_DISCONTINIUOS)
STRONG_DISCONTINIUOS_AT_INTERFACE_CPSTATE(iint::NTuple{N,Int}) where {N}  = _DISCONTINIUOS_AT_INTERFACES_CPSTATE(iint, _STRONG_DISCONTINIUOS)
WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(iint::Int)  = _DISCONTINIUOS_AT_INTERFACES_CPSTATE((iint,), _WEAK_DISCONTINIUOS)
WEAK_DISCONTINIUOS_AT_INTERFACE_CPSTATE(iint::NTuple{N,Int}) where {N}  = _DISCONTINIUOS_AT_INTERFACES_CPSTATE(iint, _WEAK_DISCONTINIUOS)


function _DISCONTINIUOS_AT_INTERFACES_CPSTATE(iint::NTuple{N,Int}, _DISCONTINIUOS::Int) where {N}
    state = Int(0)
    for i in iint
        state |= (Int(1)<<(i-1))
    end
    return CPSTATE(_DISCONTINIUOS, state)
end

const LUMPED              = CELLSTATE(_LUMPED,  [CPSTATE(_LUMPED, 0)])
const LAYERED             = CELLSTATE(_LAYERED, [CPSTATE(_LAYERED, 0)])
const FULLY_DISCONTINIUOS = CELLSTATE(_FULLY_DISCONTINIUOS, [CPSTATE(_FULLY_DISCONTINIUOS, typemax(Int))])
STRONG_DISCONTINIUOS_AT_INTERFACE(iint::NTuple{N,Int}) where {N} = _DISCONTINIUOS_AT_INTERFACES(iint, _STRONG_DISCONTINIUOS)
STRONG_DISCONTINIUOS_AT_INTERFACE(iint::Int)  = _DISCONTINIUOS_AT_INTERFACES((iint,), _STRONG_DISCONTINIUOS)
WEAK_DISCONTINIUOS_AT_INTERFACE(iint::NTuple{N,Int}) where {N} = _DISCONTINIUOS_AT_INTERFACES(iint, _WEAK_DISCONTINIUOS)
WEAK_DISCONTINIUOS_AT_INTERFACE(iint::Int)  = _DISCONTINIUOS_AT_INTERFACES((iint,), _WEAK_DISCONTINIUOS)

function _DISCONTINIUOS_AT_INTERFACES(iint::NTuple{N,Int}, _DISCONTINIUOS::Int) where {N}
    state = Int(0)
    for i in iint
        state |= (Int(1)<<(i-1))
    end
    cpstate = CPSTATE(_DISCONTINIUOS, state)
    return CELLSTATE(_DISCONTINIUOS, [cpstate])
end

is_fully_discontiniuos(c::CELL_OR_CPSTATE) = (c.state == _FULLY_DISCONTINIUOS)# && c.config==2^ninterfaces-1)
is_weak_discontiniuos(c::CELL_OR_CPSTATE) = (c.state == _WEAK_DISCONTINIUOS)
is_strong_discontiniuos(c::CELL_OR_CPSTATE) = (c.state == _STRONG_DISCONTINIUOS)
is_lumped(c::CELL_OR_CPSTATE) = (c.state == _LUMPED)
is_layered(c::CELL_OR_CPSTATE) = (c.state == _LAYERED)
is_mixed(c::CELLSTATE) = (c.state == _MIXED)
has_discontinuity(c::CELL_OR_CPSTATE) = is_fully_discontiniuos(c) || is_strong_discontiniuos(c) || is_weak_discontiniuos(c)

function is_interface_active(state::CPSTATE, iint::Int)
    is_fully_discontiniuos(state) && return true
    is_lumped(state) || is_layered(state) && return false
    
    interface_bit = ((state.config >> (iint-1)) & Int(1))
    return 1 == interface_bit
end

function is_interface_active(state::CELLSTATE, iint::Int)
    is_lumped(state) && return false
    is_layered(state) && return false
    is_fully_discontiniuos(state) && return true
    
    if is_mixed(state)
        for cpstate in state.cpstates
            if is_interface_active(cpstate, iint)
                return true
            end 
        end
        return false
    end

    if is_strong_discontiniuos(state) || is_weak_discontiniuos(state)
        return is_interface_active(first(state.cpstates), iint)
    end

    error("Unreachable code...")
end

function insert_interface(state::CPSTATE, iint::Int, ninterfaces::Int)

    config = state.config | (Int(1)<<(iint-1))
    
    new_state = state.state
    if config >= 2^ninterfaces-1 #all layers active
        new_state = _FULLY_DISCONTINIUOS
        config = typemax(Int)
    else
        if new_state == _LUMPED
            new_state = _WEAK_DISCONTINIUOS
        elseif new_state == _LAYERED
            new_state = _STRONG_DISCONTINIUOS
        end
    end

    return CPSTATE(new_state, config)
end


function insert_interface(state::CELLSTATE, iint::Int, ninterfaces::Int)
    new_cpstates = copy(state.cpstates)

    if !is_mixed(state)
        new_cpstate = insert_interface(state.cpstates[1], iint, ninterfaces)
        return CELLSTATE(new_cpstate.state, [new_cpstate])
    end

    new_cpstates[1] = insert_interface(new_cpstates[1], iint, ninterfaces)
    new_state = new_cpstates[1].state
    mixed = false
    for i in 2:length(state.cpstates)

        new_cpstates[i] = insert_interface(new_cpstates[i], iint, ninterfaces)

        if new_cpstates[2].state > new_state
            new_state = new_cpstates[2].state
        end
        if new_cpstates[1].state != new_cpstates[i].state
            mixed = true
        end
    end

    new_state = mixed==true ? _MIXED : new_state
    return CELLSTATE(new_state, new_cpstates)
end

function combine_states(a::CPSTATE, b::CPSTATE, ninterfaces::Int)
    new_state = max(a.state, b.state)
    new_config = a.config | b.config

    #Check if number of active interfaces are equal to fully_discontinous state
    new_state = (new_config >= 2^ninterfaces-1) ? _FULLY_DISCONTINIUOS : new_state
    new_config   = (new_config >= 2^ninterfaces-1) ? typemax(Int) : new_config
    return CPSTATE(new_state, new_config)
end

#=function combine_states(states::AbstractVector{CELLSTATE}, ninterfaces::Int)
    new_cell_state = states[1]
    for iint in 2:length(states)
        new_cell_state = combine_states(new_cell_state, states[iint], ninterfaces)
    end
    return new_cell_state
end=#

function generate_knot_vector(state::CPSTATE, order::Int, ninterfaces::Int)
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

function get_active_basefunctions_in_layer(ilay::Int, order::Int, state::CPSTATE)
    if state.state == _LUMPED
        return 1:(order+1)
    elseif state.state == _LAYERED
        return (1:order+1) .+ (ilay-1)*(order)
    elseif is_fully_discontiniuos(state)
        return (1:order+1) .+ (ilay-1)*(order+1)
    elseif has_discontinuity(state)
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

function get_active_basefunctions_in_interface(iint::Int, order::Int, state::CPSTATE)::UnitRange{Int}
    if is_fully_discontiniuos(state)
        return (0:1) .+ (order+1)*(iint)
    elseif is_strong_discontiniuos(state)
        offset = 0
        for i in 1:iint-1
            offset += order
            if is_interface_active(state, i) #Discontiniuos
                offset += 1
            end
        end

        if is_interface_active(state, iint)
            return (0:1) .+ (offset + (order + 1))
        else
            return (0:0) .+ (offset + (order + 1))
        end
    elseif is_weak_discontiniuos(state)
        offset = 0
        for i in 1:iint-1
            if is_interface_active(state, i) 
                offset += order+1
            end
        end

        if is_interface_active(state, iint)
            return (0:1) .+ (offset + (order + 1))
        else
            return (0:order) .+ (offset + 1)
        end

    elseif is_lumped(state)
        return 1:(order+1)
    elseif is_layered(state)
        return (0:0) .+ (order*iint + 1)
    else
        error("Wrong state")
    end
end

function generate_active_layer_dofs(nlayers::Int, order::Int, dim_s::Int, nbasefunctions_inplane::Int, state::CELLSTATE)
    active_layer_dofs = [Int[] for _ in 1:nlayers]
    dof_offset = 0
    for i in 1:nbasefunctions_inplane
        cp_state = get_cpstate(state, i)
        for ilay in 1:nlayers
            for ib in get_active_basefunctions_in_layer(ilay, order, cp_state)
                for d in 1:dim_s
                    push!(active_layer_dofs[ilay], (ib-1)*dim_s + d + dof_offset)
                end
            end
        end
        
        dof_offset += ndofs_per_controlpoint(order, nlayers, nlayers-1, dim_s, cp_state)
    end
    return active_layer_dofs
end

function generate_active_interface_dofs(ninterfaces::Int, order::Int, dim_s::Int, nbasefunctions_inplane::Int, state::CELLSTATE)
    nlayers = ninterfaces+1

    top_active_dofs = [Int[] for _ in 1:ninterfaces]
    bot_active_dofs = [Int[] for _ in 1:ninterfaces]
    active_interface_dofs = [Int[] for _ in 1:ninterfaces]

    #active_inplane_basefunction = [Int[] for _ in 1:ninterfaces]
    dof_offset = 0
    for iinpf in 1:nbasefunctions_inplane
        cp_state = get_cpstate(state, iinpf)
        for iint in 1:ninterfaces
            for ib in get_active_basefunctions_in_interface(iint, order, cp_state)
                for d in 1:dim_s
                    push!(active_interface_dofs[iint], (ib-1)*dim_s + d + dof_offset)
                end
            end
        end
        dof_offset += ndofs_per_controlpoint(order, nlayers, ninterfaces, dim_s, cp_state)
    end

    return active_interface_dofs#, active_inplane_basefunction
end

function ndofs_per_controlpoint(ooplane_order, nlayers, ninterfaces, dim_s, state::CPSTATE)
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
struct IGAShellData{dim_p,dim_s,T,LM<:Five.AbstractMaterial,IM<:Five.AbstractCohesiveMaterial}
    layer_materials::Vector{LM}
    interface_material::IM
    orders::NTuple{dim_s,Int}
    knot_vectors::NTuple{dim_p,Vector{T}}
    thickness::T
    width::T                                
    nlayers::Int                            
    zcoords::Vector{T}                      
    initial_cellstates::Vector{CELLSTATE}
    initial_interface_damages::Matrix{T}    
    adaptable::Bool                         
    limit_stress_criterion::T         
    limit_damage_criterion::T     
    search_radius::T
    locked_elements::Vector{Int}
    small_deformations_theory::Bool         
    nqp_inplane_order::Int
    nqp_ooplane_per_layer::Int
    nqp_interface_order::Int                  
    add_czcells_vtk::Bool
end

function IGAShellData(;
    layer_materials::Vector{LM},
    interface_material::IM,
    orders::NTuple{dim_s,Int},
    knot_vectors::NTuple{dim_p,Vector{T}},
    thickness::T,
    initial_cellstates::Vector{CELLSTATE},
    nqp_inplane_order::Int,
    nqp_ooplane_per_layer::Int,
    width::T                                = 1.0, #Only used in 2d,
    nlayers::Int                            = length(layer_materials),
    zcoords::Vector{T}                      = collect(-thickness/2:(thickness/nlayers):thickness/2),
    initial_interface_damages::Matrix{T}    = zeros(Float64, nlayers-1, length(initial_cellstates)),
    adaptable::Bool                         = false,
    limit_stress_criterion::T               = nothing,
    limit_damage_criterion::T               = nothing,
    search_radius::T                        = nothing,
    locked_elements::AbstractVector{Int}    = Int[],
    small_deformations_theory::Bool         = false,
    nqp_interface_order::Int                = nqp_inplane_order,
    add_czcells_vtk::Bool                   = true) where {dim_p,dim_s,T,LM<:Five.AbstractMaterial,IM<:Five.AbstractCohesiveMaterial}

    if adaptable
        if (limit_stress_criterion === nothing ||
            limit_damage_criterion === nothing ||
            search_radius === nothing ||
            locked_elements === nothing)

            error("If adaptive IGASHELL is defined, you must specify all variables related to adaptivity.")
        end
    end

    #-----
    ninterfaces = nlayers-1
    ncells = length(initial_cellstates)
    @assert(size(initial_interface_damages)[1] == ninterfaces)
    @assert(size(initial_interface_damages)[2] == ncells)
    dim_s == 3 && @assert(width == 1.0)

    return IGAShellData(layer_materials, interface_material, orders, knot_vectors, thickness, width, nlayers, zcoords, initial_cellstates, initial_interface_damages, adaptable, limit_stress_criterion, limit_damage_criterion, search_radius, collect(locked_elements), small_deformations_theory, nqp_inplane_order, nqp_ooplane_per_layer, nqp_interface_order, add_czcells_vtk)

end

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

LIMIT_STRESS_CRITERION(data::IGAShellData) = data.limit_stress_criterion
LIMIT_DAMAGE_VARIABLE(data::IGAShellData) = data.limit_damage_criterion
PROPAGATION_SEARCH_RADIUS(data::IGAShellData) = data.search_radius