struct LayerQuadratureRule{dim,T}
    qrs::Vector{QuadratureRule{dim,RefCube,T}}
end

function LayerQuadratureRule(zcoords::Vector{T}, nqp_per_layer::Int) where {T}
    nlayers = length(zcoords)-1

    qrs = QuadratureRule{1,RefCube,T}[]
    oqr = QuadratureRule{1,RefCube}(nqp_per_layer)

    addon = (last(zcoords) + first(zcoords))/2
    scale = (last(zcoords) - first(zcoords))/2
    zcoords = (zcoords.-addon)/scale
    for ilay in 1:nlayers
        layer_qr = QuadratureRule{1,RefCube,T}(T[],Vec{1,T}[])

        addon = (zcoords[ilay+1] + zcoords[ilay])/2
        scale = (zcoords[ilay+1] - zcoords[ilay])/2
        for qp in 1:length(oqr.weights)
            new_z = oqr.points[qp]*scale .+ addon
            push!(layer_qr.points,  Vec(Tuple(new_z)))
            push!(layer_qr.weights, oqr.weights[qp]*scale)
        end

        push!(qrs, layer_qr)
    end

    return LayerQuadratureRule(qrs)
end

nlayers(qr::LayerQuadratureRule) = length(qr.qrs)
getnweights(qr::LayerQuadratureRule) = sum([length(qr.qrs[i].weights) for i in 1:nlayers(qr)])
Base.getindex(qr::LayerQuadratureRule, i::Int) = qr.qrs[i]

function combine_qrs(qr_inplane::QuadratureRule{dim_p}, qr_ooplane::LayerQuadratureRule{1,T}) where {dim_p,T}
    dim_s = dim_p+1
    qrs = QuadratureRule{dim_s,RefCube,T}[]

    for ilay in 1:nlayers(qr_ooplane)
        layer_qr = QuadratureRule{dim_s,RefCube,T}(T[],Vec{dim_s,T}[])

        for oqp in 1:length(qr_ooplane.qrs[ilay].weights)
            for iqp in 1:length(qr_inplane.weights)
                ξη = qr_inplane.points[iqp]
                ζ = qr_ooplane.qrs[ilay].points[oqp]
                _p = [ξη..., ζ...]
                _w = qr_inplane.weights[iqp] * qr_ooplane.qrs[ilay].weights[oqp]
                push!(layer_qr.points, Vec{dim_s,T}(Tuple(_p)))
                push!(layer_qr.weights, _w)
            end
        end
        push!(qrs, layer_qr)
    end
    
    return LayerQuadratureRule(qrs)
end

function generate_face_oop_qrs(nlayers::Int)
    T = Float64
    qr_bot = QuadratureRule{1,RefCube,T}(T[1.0], [Vec((-1.0,))])
    qr_top = QuadratureRule{1,RefCube,T}(T[1.0], [Vec((+1.0,))])

    qrs_bot = [QuadratureRule{1,RefCube,T}(T[], Vec{1,T}[]) for i in 1:nlayers]
    qrs_top = [QuadratureRule{1,RefCube,T}(T[], Vec{1,T}[]) for i in 1:nlayers]

    qrs_bot[1] = qr_bot
    qrs_top[nlayers] = qr_top

    return [LayerQuadratureRule(qrs_bot), LayerQuadratureRule(qrs_top)]
end

function generate_cohesive_oop_qrs(zcoords::Vector{T}) where {T}
    
    nlayers = length(zcoords)-1
    ε = 1e-13

    qrs_top = QuadratureRule{1,RefCube,T}[]
    qrs_bot = QuadratureRule{1,RefCube,T}[]
    
    #Change to -1 to 1
    zcoords_interfaces = change_zcoord_range(zcoords)

    #Top
    push!(qrs_top, QuadratureRule{1,RefCube,T}(T[],Vec{1,T}[]))
    for ilay in 2:nlayers
        layer_qr = QuadratureRule{1,RefCube,T}(T[],Vec{1,T}[])

        #Add the zcoord to the quadrature rule
        push!(layer_qr.points,  Vec( (zcoords_interfaces[ilay] + ε, ) ))
        push!(layer_qr.weights, 1.0)

        push!(qrs_top, layer_qr)
    end

    #bot
    for ilay in 2:nlayers
        layer_qr = QuadratureRule{1,RefCube,T}(T[],Vec{1,T}[])

        #Add the zcoord to the quadrature rule
        push!(layer_qr.points,  Vec( (zcoords_interfaces[ilay] - ε, ) ))
        push!(layer_qr.weights, 1.0)
        
        push!(qrs_bot, layer_qr)
    end
    push!(qrs_bot, QuadratureRule{1,RefCube,T}(T[],Vec{1,T}[]))

    return [LayerQuadratureRule(qrs_bot), LayerQuadratureRule(qrs_top)]

end

function generate_plot_oop_qr(zcoords::Vector{T}) where {T}
    
    nlayers = length(zcoords)-1
    ε = 1e-13
    qrs = QuadratureRule{1,RefCube,T}[]

    #Change to -1 to 1
    zcoords = change_zcoord_range(zcoords)
    for ilay in 1:nlayers
        layer_qr = QuadratureRule{1,RefCube,T}(T[],Vec{1,T}[])

        #Add the zcoord to the quadrature rule
        push!(layer_qr.points,  Vec( (zcoords[ilay] + ε, ) ))
        push!(layer_qr.points,  Vec( (zcoords[ilay+1] - ε, ) ))
        push!(layer_qr.weights, 1.0)
        push!(layer_qr.weights, 1.0)

        push!(qrs, layer_qr)
    end
    return LayerQuadratureRule(qrs)
end


#changes zcoord to -1 to 1
function change_zcoord_range(zcoords::Vector)
    addon = (last(zcoords) + first(zcoords))/2
    scale = (last(zcoords) - first(zcoords))/2
    zcoords = (zcoords.-addon)/scale
    return zcoords
end