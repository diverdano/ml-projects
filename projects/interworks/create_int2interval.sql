create or replace function udf_int2interval(integer) returns interval
  LANGUAGE sql
  AS
  $function$
    select concat($1, ' min')::interval;
  $function$
  returns null on null input;
