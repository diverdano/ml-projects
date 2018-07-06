create or replace function udf_int2time(integer) returns text
  LANGUAGE sql
  AS
  $function$
    select to_char(to_char($1, '0000')::time, 'HH24:MI');
  $function$
  returns null on null input;
