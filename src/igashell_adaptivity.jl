"""
    IGAShellAdaptivity  

Stores the cellstates and controlpoints states.
Also stores the bezier-extraction operators for getting the new dofvalues of a controlpoint state.
"""
struct IGAShellAdaptivity{T}
    cellstates::Vector{CELLSTATE}
    control_point_states::Vector{CPSTATE}
    propagation_checked::Matrix{Bool}
    locked_control_points::Vector{Int}

    lumped2layered::IGA.BezierExtractionOperator{T}
    layered2fullydiscont::IGA.BezierExtractionOperator{T}
    lumped2fullydiscont::IGA.BezierExtractionOperator{T}
    weakdiscont2fullydiscont::Dict{Int, IGA.BezierExtractionOperator{T}}
    strongdiscont2fullydiscont::Dict{Int, IGA.BezierExtractionOperator{T}}

    interface_knots::Vector{T}
    order::Int
end

function getcellstate(adap::IGAShellAdaptivity, cellid::Int)
    return adap.cellstates[cellid]
end

function get_controlpoint_state(adap::IGAShellAdaptivity, cpid::Int)
    return adap.control_point_states[cpid]
end

function get_controlpoint_state(adap::IGAShellAdaptivity, cpid::Vector{Int})
    return adap.control_point_states[cpid]
end


function set_controlpoint_state!(adap::IGAShellAdaptivity, cpid::Int, state::CPSTATE)
    adap.control_point_states[cpid] = state
end

function setcellstate!(adap::IGAShellAdaptivity, cellid::Int, state::CELLSTATE)
    adap.cellstates[cellid] = state
end

function IGAShellAdaptivity(data::IGAShellData{dim_p,dim_s,T}, cell_connectivity::Matrix{Int}, ncells, nnodes) where {dim_p,dim_s,T}

    order = data.orders[dim_s]
    ninterfaces = data.nlayers-1
    knot_lumped = generate_knot_vector(order, ninterfaces, 0)

    #lumped to layered
    interface_knots = [-1 + 2i/(data.nlayers) for i in 1:(ninterfaces)]
    Cmat_lu2la = generate_out_of_plane_extraction_operators(knot_lumped, order, interface_knots, fill(order, length(interface_knots)))

    #layered to discont
    knot_layered = generate_knot_vector(order, ninterfaces, order)
    Cmat_la2di = generate_out_of_plane_extraction_operators(knot_layered, order, interface_knots, fill(1, length(interface_knots)))

    #lumped to discont
    Cmat_lu2di = Cmat_la2di*Cmat_lu2la

    #Discont to discont
    weakdiscont2discont = Dict{Int, IGA.BezierExtractionOperator{T}}()
    strongdiscont2discont = Dict{Int, IGA.BezierExtractionOperator{T}}()

    Clu2la = IGA.bezier_extraction_to_vector(Cmat_lu2la')
    Cla2di = IGA.bezier_extraction_to_vector(Cmat_la2di')
    Clu2di = IGA.bezier_extraction_to_vector(Cmat_lu2di')

    #Some of the cells will be initialized with LUMPED, and some with LYARED/DISCONTINIUOS
    # This means that some cells will be in a mixed mode... Determine those
    # Prioritize the DISCONTINIUOS before LAYERED and LUMPED, ie. if one node is both lumped 
    # and disocontinous, choose it to be dinscontionous
    node_states = fill(LUMPED_CPSTATE, nnodes)

    for cellid in 1:ncells
        cellstate = data.initial_cellstates[cellid] 
        for cellnodes in cell_connectivity[:, cellid]
            for nodeid in cellnodes
                node_states[nodeid] = combine_states(node_states[nodeid], first(cellstate.cpstates), ninterfaces)
            end
        end
    end

    #Check if there is any cell that is has MIXED states controlpoints
    for cellid in 1:ncells
        nodeids = cell_connectivity[:, cellid]
        cellnode_states = node_states[nodeids]

        #if NOT all of the nodes in the cell are equal, the element is mixed
        if !all(first(cellnode_states) .== cellnode_states)
            data.initial_cellstates[cellid] = CELLSTATE(_MIXED, copy(cellnode_states))
        end

    end
    
    propagation_checked = [false for _ in 1:ninterfaces, _ in 1:ncells]

    #Some cells are not allowed to be upgraded (locked).. store the contorl points of these cells
    locked_control_points = Int[]
    for cellid in data.locked_elements
        for cell_nodes in cell_connectivity[:, cellid]
            for nodeid in cell_nodes
                push!(locked_control_points, nodeid)
            end
        end
    end

    return IGAShellAdaptivity(data.initial_cellstates, node_states, propagation_checked, locked_control_points,
                              Clu2la, Cla2di, Clu2di, 
                              weakdiscont2discont, strongdiscont2discont,
                              interface_knots, order)
end

function get_upgrade_operator(adap::IGAShellAdaptivity, from::CPSTATE, to::CPSTATE)
    ninterfaces = length(adap.interface_knots)

    if is_lumped(from) && is_layered(to)
        return adap.lumped2layered
    elseif is_lumped(from) && is_fully_discontiniuos(to)
        return adap.lumped2fullydiscont
    elseif is_layered(from) && is_fully_discontiniuos(to)
        return adap.layered2fullydiscont
    elseif is_weak_discontiniuos(from) && is_fully_discontiniuos(to)
        return _get_upgrade_operator(adap.weakdiscont2fullydiscont, adap.order, adap.interface_knots, from, to)
    elseif is_strong_discontiniuos(from) && is_fully_discontiniuos(to)
        return _get_upgrade_operator(adap.strongdiscont2fullydiscont, adap.order, adap.interface_knots, from, to)
    elseif (is_layered(from) || is_lumped(from)) && has_discontinuity(to)
        return create_upgrade_operator(adap.order, adap.interface_knots, from, to)
    elseif has_discontinuity(from) && has_discontinuity(to)
        return create_upgrade_operator(adap.order, adap.interface_knots, from, to)
    else
        error("Wrong upgrade, $from -> $to")
    end

end

function _get_upgrade_operator(dict::Dict, order::Int, interface_knots::Vector{Float64}, from::CPSTATE, to::CPSTATE)
    @assert(has_discontinuity(from) && is_fully_discontiniuos(to))

    return get!(dict, from.config) do
        return create_upgrade_operator(order, interface_knots, from, to)
    end
end

function create_upgrade_operator(order::Int, interface_knots::Vector{Float64}, from::CPSTATE, to::CPSTATE)
    
    ninterfaces = length(interface_knots)
    kv = generate_knot_vector(from, order, ninterfaces)

    new_state = combine_states(from, to, ninterfaces)
    
    from_multiplicity = generate_nmultiplicity_vector(from, ninterfaces, order)
    new_multiplicity = generate_nmultiplicity_vector(new_state, ninterfaces, order)
    upgrade_multiplicity = new_multiplicity - from_multiplicity

    C = generate_out_of_plane_extraction_operators(kv, order, interface_knots, upgrade_multiplicity)
    return IGA.bezier_extraction_to_vector(C')
end