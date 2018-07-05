drop view if exists vw_next_day;
drop view if exists vw_flights;
drop view if exists vw_flights4analysis;
drop view if exists vw_next_day;

create view vw_flights as select
    id_trans,
    date_flight::date,
    id_airline,   -- join and do upper()
    id_tailnum,
    id_flightnum,
    id_airport_orig, -- join and do upper(),
    id_airport_dest, -- join and do upper(),
    udf_int2time(time_depart_crs) as t_depart_crs,
    udf_int2time(time_depart) as t_depart,
    udf_int2interval(time_depart_delay) as i_depart_delay,
    udf_int2time(time_taxi_out) as t_taxi_out,
    udf_int2time(time_wheelsoff) as t__wheelsoff,
    udf_int2time(time_wheelson) as t_wheelson,
    udf_int2time(time_taxi_in) as t_taxi_in,
    udf_int2time(time_arrive_crs) as t_arrive_crs,
    udf_int2time(time_arrive) as t_arrive,
    udf_int2interval(time_arrive_delay),
    udf_int2interval(time_elapsed_crs) as i_elapsed_crs,
    udf_int2interval(time_elapsed_act) as i_elapsed_act,
    stat_cancelled,
    stat_diverted,
    stat_miles
    from flights;

create view vw_next_day as select
    id_trans,
    date_flight,
    id_airline,   -- join and do upper()
    id_tailnum,
    id_flightnum,
    t_depart,
    t_arrive,
    i_elapsed_act,
    i_elapsed_crs,
    (case when i_elapsed_crs::interval > t_arrive::interval then 'y' else 'n' end) as next_day
    from vw_flights;

create view vw_flights4analysis as select
    id_trans,
    date_flight::date,
    fl.id_airline,   -- join and do upper()
    al.name as airline_name,
    id_tailnum,
    id_flightnum,
    id_airport_orig, -- join and do upper(),
    id_airport_dest, -- join and do upper(),
    apo.name as airport_orig_name,
    apo.city as airport_orig_city,
    apo.st as airport_orig_st,
    apd.name as airport_dest_name,
    apd.city as airport_dest_city,
    apd.st as airport_dest_st,
    udf_int2time(time_depart_crs) as t_depart_crs,
    udf_int2time(time_depart) as t_depart,
    udf_int2interval(time_depart_delay) as i_depart_delay,
    (case when udf_int2interval(time_depart_delay) > '00:15'::interval then 'y' else 'n' end) as delay_gt_15,
    udf_int2time(time_taxi_out) as t_taxi_out,
    udf_int2time(time_wheelsoff) as t__wheelsoff,
    udf_int2time(time_wheelson) as t_wheelson,
    udf_int2time(time_taxi_in) as t_taxi_in,
    udf_int2time(time_arrive_crs) as t_arrive_crs,
    udf_int2time(time_arrive) as t_arrive,
    udf_int2interval(time_arrive_delay),
    udf_int2interval(time_elapsed_crs) as i_elapsed_crs,
    udf_int2interval(time_elapsed_act) as i_elapsed_act,
    stat_cancelled,
    stat_diverted,
    stat_miles,
    round(stat_miles/100)*100 || '-' ||(1+round(stat_miles/100))*100|| ' miles' as distance_group,
    (case when udf_int2interval(time_elapsed_crs)::interval > udf_int2time(time_arrive)::interval then 'y' else 'n' end) as next_day
    from flights fl
    left join airlines al on fl.id_airline = al.id_airline
    left join airports apo on fl.id_airport_orig = apo.id_airport
    left join airports apd on fl.id_airport_dest = apd.id_airport;
