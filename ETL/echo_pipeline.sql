/****** Script for SelectTopNRows command from SSMS  ******/

-----------MITRAL REGURGITATION PIPELINE---------------------------------------------------------------------------
;with cte_mr_1 as(
select *,
CASE when a.Findings_Name like '%mitralvalve%' then

                    case
					     when a.Findings_Value like '%mitral regurgitation%'  OR a.Findings_Value COLLATE Latin1_General_BIN LIKE'% MR %' then
				      case
	                       when a.findings_value like 'A trace of mitral regurgitation.' 
				 or a.findings_value like 'mitral regurgitation.'
				 or a.findings_value like 'there is mitral regurgitation.'
                 or a.findings_value like 'A trace of continous mitral regurgitation.'
                 or a.findings_value like 'A trace of eccentric mitral regurgitation.'
				 or a.findings_value like '%possible mitral regurgitation present%'
				 or a.findings_value like '%continuous mitral regurgitation%'
				 or a.findings_value like '%Continous mitral regurgitation%'
				 or a.findings_value like '%Continnous mitral regurgitation%'
				 or a.findings_value like 'mitral regurgitation present'
                 or a.findings_value like 'A trace of central mitral regurgitation.'
                 or a.findings_value like 'A trace of valvular mitral regurgitation.'
                 or a.findings_value like 'A trace of posteriorly directed mitral regurgitation.'
                 or a.findings_value like '%trace%' 
				 or a.findings_value like '%trivial%'
				 or a.findings_value like '%Mitral regurgitation present; severity uncertain.%'
				 or a.findings_value like '%least itral regurgitation present%'
				 or a.findings_value like '%Eccentric mitral regurgitation present'
				 or a.findings_value like 'mitral regurgitation is present.'
				 or a.findings_value like 'Continuous mitral regurgitation.' 
				 and a.findings_value not like '%moderate%'
				 and a.findings_value not like '%severe%'

				                                               then 'mild mitral regurgitation'
	                       when a.Findings_Value like 'Clinically insignificant mitral regurgitation.' 
				 or a.Findings_Value like '%Clinically insignificant mitral regurgitation%'
				 or a.Findings_Value like '%no sig%'
							                                   then 'mild mitral regurgitation'----MILD-------

		                    when a.Findings_Value like 'mild mitral regurgitation.' 
				 or a.Findings_Value like '%mild mitral regurgitation%'
				 or a.Findings_Value like '%mild central%'
				 or a.Findings_Value like '%mild late%'
				 or a.Findings_Value like '%mild  late%'
				 or a.Findings_Value like '%mild early%'
				 or a.Findings_Value like '%mild eccentric%'
				 or a.Findings_Value like '%mild calcific%'
				 or a.Findings_Value like '%mild non-holosystolic%'
				 or a.Findings_Value like '%mild residual%'
				 or a.Findings_Value like '%Mild  mitral regurgitation.%'
				 or a.Findings_Value like '%Mild mid%'
				 or a.Findings_Value like '%minimal%'
				 or a.Findings_Value like '%mild%'
				and a.Findings_Value not like '%mild to moderate%'
				and a.Findings_Value not like '%mild tp moderate%'
				and a.Findings_Value not like '%mild-to-moderate%'
				and a.Findings_Value not like '%mild-tp-moderate%'
				and a.Findings_Value not like '%mild-moderate%'
				and a.Findings_Value not like '%trace-mild%'
				and a.Findings_Value not like '%trace to mild%'
				and a.Findings_Value not like '%trace/mild%' 
								                            then 'mild mitral regurgitation'
		           when a.Findings_Value like 'Mild to moderate mitral regurgitation.' 
				or a.Findings_Value like '%Mild to moderate%'
				or a.Findings_Value like '%Mild tp moderate%'
				or a.Findings_Value like '%Mild-tp-moderate%'
				or a.Findings_Value like '%Mild-to-moderate%'
				or a.Findings_Value like '%Mild-moderate%'
								                          then 'Mild to moderate mitral regurgitation'
		            when a.Findings_Value like 'moderate mitral regurgitation.' 
								   or a.Findings_Value like '%moderate mitral regurgitation%'
								   or a.Findings_Value like '%modate%'
								   or a.Findings_Value like '%mderate%'
								   or a.Findings_Value like '%moderate%'
								   and a.Findings_Value not like '%mild to moderate%'
								   and a.Findings_Value not like '%mild tp moderate%'
								   and a.Findings_Value not like '%mild-to-moderate%'
								   and a.Findings_Value not like '%mild-tp-moderate%'
								   and a.Findings_Value not like '%mild-moderate%'
								   and a.Findings_Value not like '%mild moderate%'
								   and a.Findings_Value not like '%mild/moderate%'
								   and a.Findings_Value not like '%mild%'
								   and a.Findings_Value not like '%mod-severe%'
								   and a.Findings_Value not like '%severe%'
								                            then 'moderate mitral regurgitation'
		            when a.Findings_Value like '%moderate to severe mitral regurgitation%' 
								 or a.Findings_Value like '%moderate-severe%'
								 or a.Findings_Value like '%moderate-to-severe%'
								 or a.Findings_Value like '%moderate-tp-severe%'
								 or a.Findings_Value like '%moderate to severe%'
								 or a.Findings_Value like '%moderate tp severe%'
								 or a.Findings_Value like '%moderate or severe%'
								 or a.Findings_Value like '%moderately severe%'
								 or a.Findings_Value like '%mod-severe%'
								 or a.Findings_Value like '%moderate but cannot exclude severe%'
								 or a.Findings_Value like '%moderate  to severe  regurgitation%'

								                                            then 'moderate to severe mitral regurgitation'
		          when a.Findings_Value like '%severe mitral regurgitation%'
								 or a.Findings_Value like '%severe%'
								 and a.Findings_Value not like 'moderate to severe mitral regurgitation.' 
								 and a.Findings_Value not like '%moderate-severe%'
								 and a.Findings_Value not like '%moderate-to-severe%'
								 and a.Findings_Value not like '%moderate-tp-severe%'
								 and a.Findings_Value not like '%moderate to severe%'
								 and a.Findings_Value not like '%moderate  to severe%'
								 and a.Findings_Value not like '%moderate tp severe%'
								 and a.Findings_Value not like '%moderately severe%'
								 
								                                             then 'severe mitral regurgitation'																	                                  
                  when a.Findings_Value like '%Cannot exclude mitral regurgitation%'--make this indeterminate--
				  or a.Findings_Value like '%appears to be mitral regurgitation%'--make this indeterminate--
				  or a.Findings_Value like '%cannot rule out%'--make this indeterminate--
				  --or a.Findings_Value like '%jet%'--take this out---
				                                                then 'indeterminate'--
				 when a.Findings_Value like '%no evidence of mitral regurgitation%'
				 or a.Findings_Value like '%no evidence of%'
				 or a.Findings_Value like 'Mitral valve not well seen. No mitral regurgitation.'
				                                          then 'NA
else 'indeterminate'
end
else 'NA'
end
end as 'MR_severity'
from ECHO_FULLDATASET a)
,cte_mr_2 as(
select * from cte_mr_1
where MR_severity<>'NA' and MR_severity is not null)
,CTE_MR_3 AS (
select DISTINCT
ORDERID,PatientID,StudyDate,MR_severity
from cte_mr_1
where orderid in (select distinct orderid from cte_mr_2)
and MR_severity<>'NA' and MR_severity is not null)
,CTE_MR_4 AS (
select *,
case when MR_severity='severe mitral regurgitation' then '1'
when MR_severity='moderate to severe mitral regurgitation' then '2'
when MR_severity='moderate mitral regurgitation' then '3'
when MR_severity='Mild to moderate mitral regurgitation' then '4'
when MR_severity='mild mitral regurgitation' then '5'
when MR_severity='indeterminate' then '6'
end as MR_severity_priority
from CTE_MR_3)
,CTE_MR_5 AS (
SELECT * FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC,MR_severity_priority ASC
  ) as row_num FROM CTE_MR_4 )LOOP WHERE LOOP.row_num=1)
,cte_mr_6 as (
SELECT * FROM cte_mr_1
WHERE ORDERID IN (
select distinct orderid from cte_mr_1
except
select distinct orderid from cte_mr_2))
,cte_mr_7 as (
select distinct orderid,patientid,studydate
from cte_mr_6)
,cte_mr_8 as (
select * from (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC
  ) as row_num FROM cte_mr_7)loop where loop.row_num=1)
 ,cte_mr_9 as (
select distinct orderid,patientid,studydate,
case when orderid in (select distinct orderid from cte_mr_8) then 'NA' end as MR_severity
from cte_mr_8
union
select distinct orderid,patientid,studydate,MR_severity
from CTE_MR_5)

---TEMP TABLE FOR MITRAL REGURGITATION SEVERITY N=175,399
Select * into #tmpmr9 from cte_mr_9

DROP TABLE IF EXISTS MITRAL_REGURGITATION_FULLDATASET_PATIENT
SELECT * INTO MITRAL_REGURGITATION_FULLDATASET_PATIENT
FROM #tmpmr9

----------PIPELINE FOR MITRAL STENOSIS----------------------------------------------
;WITH CTE_MS_1 AS (select *,
CASE when a.Findings_Name like '%mitralvalve%' or Findings_Name like '%FINDINGS_DopplerFindingsDiastology%' then

case
    when a.Findings_Value like '%mitral stenosis%' OR A.Findings_Value LIKE '% MS %' AND A.Findings_Value NOT LIKE '%[0-9] MS %'
	and a.Findings_Value not like '%mitral regurgitation%' 
	and a.Findings_Value not like '% MR %' 
	then
		   case
		       when a.findings_value like 'A trace of mitral stenosis.'
	           or a.findings_value like '%trivial%'
			   or a.Findings_Value like '%Unable to assess diastolic function due to MS and severe MAC%'
			                                       then 'Mild mitral stenosis'
	           when a.Findings_Value like 'insignificant mitral stenosis.' 
		         or a.Findings_Value like '%insignificant%'
				 or a.Findings_Value like '%no significant mitral%'
				 or a.Findings_Value like '%no significant  mitral%'
				 or a.Findings_Value like '%no  significant mitral%'
				 
				                                       then 'Mild mitral stenosis'
		       when a.Findings_Value like '%mild mitral stenosis%' 
		   or a.Findings_Value like '%mild calci%'
		   or a.Findings_Value like '%mild  calci%'
		   or a.Findings_Value like '%mild  mitral%'
		   or a.Findings_Value like '%mild (calcific)%'
		   or a.Findings_Value like '%mild clacific%'
		   or a.Findings_Value like '%mild residual%'
		   or a.Findings_Value like '%mild rheumatic%'
		   or a.Findings_Value like '%mild inflow%'
		   or a.Findings_Value like '%mild low%'
		   or a.Findings_Value like '%minimal%'
		   or a.Findings_Value like '%midl%'
		   and a.Findings_Value not like '%mild to moderate%'
		   and a.findings_value not like '%Mild to moderate mitral stenosis%' 
		   and a.findings_value not like '%mild-moderate%'
		   and a.findings_value not like '%mild-mod%'  then 'mild mitral stenosis'
		   when a.Findings_Value like 'Mild to moderate mitral stenosis.'
		   or a.Findings_Value like '%Mild to moderate%'
		   or a.Findings_Value like '%Mild-to-moderate%'
		   or a.Findings_Value like '%Mild tp moderate%'
		   or a.Findings_Value like '%Mild-tp-moderate%'
		   or a.Findings_Value like '%Mild to moserate%'
		   or a.Findings_Value like '%Mild-moderate%'
		   or a.Findings_Value like '%Mild-mod%' then 'Mild to moderate mitral stenosis'
		   when a.Findings_Value like '%moderate mitral stenosis%'
		   or a.Findings_Value like '%moderate%'
		   and  a.Findings_Value not like  '%mild%'
		   and  a.Findings_Value not like  '%severe%' then 'moderate mitral stenosis'
		   when a.Findings_Value like 'moderate to severe mitral stenosis.'
		   or a.Findings_Value like '%moderate-severe%'
		   or a.Findings_Value like '%moderately severe%'
		   or a.Findings_Value like '%moderate to severe%'
		   or a.Findings_Value like '%moderate  to severe%'
		   or a.Findings_Value like '%moderate-to-severe%'
		   or a.Findings_Value like '%moderate-tp-severe%' 
		   or a.Findings_Value like '%moderate tp severe%' then 'moderate to severe mitral stenosis'
		   when a.Findings_Value like 'severe mitral stenosis.'
		   or a.Findings_Value like '%severe%'  
		   and a.Findings_Value not like '%moderate%' then 'severe mitral stenosis'
		   when a.Findings_Value like 'no evidence of mitral stenosis.'
		   or a.Findings_Value like '%no evidence%' then 'NA'
		   when a.Findings_value like 'Cannot exclude mitral stenosis.'
		        or a.Findings_value like '%Cannot exclude%' then 'indeterminate'
else 'indeterminate'
end
else 'NA'
end
end as 'MS_severity'
from NASIR.DBO.ECHO_FULLDATASET a)
,CTE_MS_2 
as(select * from CTE_MS_1
where MS_severity<>'NA' and MS_severity is not null)
,CTE_MS_3 AS (
select DISTINCT
ORDERID,PatientID,StudyDate,MS_severity
from CTE_MS_1
where orderid in (select distinct orderid from CTE_MS_2)
and MS_severity<>'NA' and MS_severity is not null)
,CTE_MS_4 AS (
select *,
case when MS_severity='severe mitral stenosis' then '1'
when MS_severity='moderate to severe mitral stenosis' then '2'
when MS_severity='moderate mitral stenosis' then '3'
when MS_severity='Mild to moderate mitral stenosis' then '4'
when MS_severity='mild mitral stenosis' then '5'
when MS_severity='indeterminate' then '6'
end as MS_severity_priority
from CTE_MS_3)
,CTE_MS_5 AS (
SELECT * FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC,MS_severity_priority ASC
  ) as row_num FROM CTE_MS_4 )LOOP WHERE LOOP.row_num=1)
,CTE_MS_6 as (
SELECT * FROM CTE_MS_1
WHERE ORDERID IN (
select distinct orderid from CTE_MS_1
except
select distinct orderid from CTE_MS_2))
,CTE_MS_7 as (
select distinct orderid,patientid,studydate
from CTE_MS_6)
,CTE_MS_8 as (
select * from (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC
  ) as row_num FROM CTE_MS_7)loop where loop.row_num=1)
,CTE_MS_9 as (
select distinct orderid,patientid,studydate,
case when orderid in (select distinct orderid from CTE_MS_8) then 'NA' end as MS_severity
from CTE_MS_8
union
select distinct orderid,patientid,studydate,MS_severity
from CTE_MS_5)
---TEMP TABLE FOR MITRAL STENOSIS SEVERITY N=175,399
Select * into #tmpmss9 from CTE_MS_9
DROP TABLE IF EXISTS MITRAL_STENOSIS_FULLDATASET_PATIENT
SELECT * INTO MITRAL_STENOSIS_FULLDATASET_PATIENT
FROM #tmpmss9



-------PIPELINE FOR AORTIC REGIRGITATION------------------------
;with cte_ar_1 as (select *,
CASE when a.Findings_Name like '%aorticvalve%' then
case
    when a.Findings_Value like '%aortic regurgitation%' or a.Findings_Value like '% AR %' then
                  case
			           when a.findings_value like 'A trace of aortic regurgitation.'
				       or a.findings_value like 'aortic regurgitation.'
                       or a.findings_value like 'there is aortic regurgitation.'
                       or a.findings_value like 'A trace of continous aortic regurgitation.'
                       or a.findings_value like 'A trace of eccentric aortic regurgitation.'
                       or a.findings_value like 'eccentric aortic regurgitation.'
				       or a.findings_value like '%possible aortic regurgitation present%'
				       or a.findings_value like '%continuous aortic regurgitation%'
                       or a.findings_value like '%Continous aortic regurgitation%'
				       or a.findings_value like '%Continnous aortic regurgitation%'
				       or a.findings_value like 'aortic regurgitation present%'
                       or a.findings_value like 'A trace of central aortic regurgitation.'
                       or a.findings_value like 'A trace of valvular aortic regurgitation.'
                       or a.findings_value like 'A trace of posteriorly directed aortic regurgitation.'
                       or a.findings_value like '%trace%' 
				       or a.findings_value like '%trivial%'
				       or a.findings_value like 'Aortic regurgitation is present.'
				       or a.findings_value like 'Continuous aortic regurgitation.' 
				       or a.findings_value like '%Aortic regurgitation is present%'
				                                                                   then 'Mild aortic regurgitation'
                        when a.Findings_Value like 'No evidence of aortic regurgitation.'
                          or a.findings_value like 'No evidence of continous aortic regurgitation.'
                          or a.findings_value like 'No evidence of eccentric aortic regurgitation.'
                          or a.findings_value like 'No evidence of central aortic regurgitation.'
                          or a.findings_value like 'No evidence of valvular aortic regurgitation.'
                          or a.findings_value like 'No evidence of posteriorly directed aortic regurgitation.'
                          or a.findings_value like '%no%evidence%'
				          or a.findings_value like '%uncertain%'
				          or a.findings_value like '%no aortic regurgitation%'
				                                                              then 'NA'
                          when a.Findings_Value like 'mild aortic regurgitation.'
                          or a.findings_value like 'mild continous aortic regurgitation.'
                          or a.findings_value like 'mild eccentric aortic regurgitation.'
                          or a.findings_value like 'mild central aortic regurgitation.'
                          or a.findings_value like 'mild valvular aortic regurgitation.'
                          or a.findings_value like 'mild posteriorly directed aortic regurgitation.' 
                          or a.findings_value like '%mild%' 
		                  and a.findings_value not like '%mild to moderate%' 
		                  and a.findings_value not like '%mild-moderate%'
		                  and a.findings_value not like '%mild-mod%' 
				                                                             then 'mild aortic regurgitation'

                         when a.Findings_Value like 'Mild to moderate aortic regurgitation.'
                          or a.findings_value like 'Mild to moderate continous aortic regurgitation.'
                          or a.findings_value like 'Mild to moderate eccentric aortic regurgitation.'
                          or a.findings_value like 'Mild to moderate central aortic regurgitation.'
                          or a.findings_value like 'Mild to moderate valvular aortic regurgitation.'
                          or a.findings_value like 'Mild to moderate posteriorly directed aortic regurgitation.'
                          or a.findings_value like '%mild to moderate%'
		                  or a.findings_value like '%mild-moderate%' 
				                                                    then 'Mild to moderate aortic regurgitation'

                          when a.Findings_Value like 'moderate aortic regurgitation.'
                           or a.findings_value like 'moderate continous aortic regurgitation.'
                           or a.findings_value like 'moderate eccentric aortic regurgitation.'
                           or a.findings_value like 'moderate central aortic regurgitation.'
                           or a.findings_value like 'moderate valvular aortic regurgitation.'
                           or a.findings_value like 'moderate posteriorly directed aortic regurgitation.'
                           or a.findings_value like '%moderate%' 
		                   and a.findings_value not like '%moderate to severe%' 
		                   and a.findings_value not like '%moderate-severe%'
		                   and a.findings_value not like '%mod-severe%'
		                   and a.findings_value not like '%mild to moderate%'
		                   and a.findings_value not like '%mild-moderate%'
		                   and a.findings_value not like '%mild-mod%' 
				                                                    then 'moderate aortic regurgitation'

                          when a.Findings_Value like 'moderate to severe aortic regurgitation.'
                          or a.findings_value like 'moderate to severe continous aortic regurgitation.'
                          or a.findings_value like 'moderate to severe eccentric aortic regurgitation.'
                          or a.findings_value like 'moderate to severe central aortic regurgitation.'
                          or a.findings_value like 'moderate to severe valvular aortic regurgitation.'
                          or a.findings_value like 'moderate to severe posteriorly directed aortic regurgitation.' 
                          or a.findings_value like '%moderate to severe%'
		                  or a.findings_value like '%moderate-severe%' 
				          or a.findings_value like '%moderatly severe%'
				          or a.findings_value like '%moderately severe%'
		                  or a.findings_value like '%mod-severe%' 
				                                                    then 'moderate to severe aortic regurgitation'

                          when a.Findings_Value like 'severe aortic regurgitation.'
                          or a.Findings_Value like 'Severe%'
			              or a.Findings_Value like 'probably severe%'
                          or a.findings_value like 'severe continous aortic regurgitation.'
                          or a.Findings_Value like 'Suspect severe eccentric %neo%aortic regurgitation.%'
	                      or a.Findings_Value like 'Suspect severe eccentric aortic regurgitation.%'
                          or a.findings_value like 'severe eccentric aortic regurgitation.'
                          or a.findings_value like 'severe central aortic regurgitation.'
                          or a.findings_value like 'severe valvular aortic regurgitation.'
                          or a.findings_value like 'severe posteriorly directed aortic regurgitation.' 
						  or a.Findings_Value like 'Cannot exclude severe aortic regurgitation.'
						  or a.Findings_Value like '%difficult to assess severity of aortic regurgitation, probably severe%'
                          and a.findings_value not like '%moderate to severe%'
		                  and a.findings_value not like '%moderate-severe%' 
		                  and a.findings_value not like '%mod-severe%'
			                                                         then 'severe aortic regurgitation'

                           when a.Findings_value like '%Cannot exclude%'
                           or a.Findings_value like '%significant%'
			               or a.Findings_value like '%Diastolic motion shows diastolic flutter consistent with aortic regurgitation.%' 
			               or a.Findings_value like '%No structural AV abnormalities noted. AV does not open with continuous aortic regurgitation%' 
			               or a.Findings_value like '%No change in the aortic regurgitation%'
			               or a.Findings_value like '%further charac%'
                                                                     then 'indeterminate'
else 'indeterminate'
end
else 'NA'
end
end as 'AR_severity'
from ECHO_FULLDATASET a)
,cte_ar_2 
as(select * from cte_ar_1
where AR_severity<>'NA' and AR_severity is not null)
,cte_ar_3 AS (
select DISTINCT
ORDERID,PatientID,StudyDate,AR_severity
from cte_ar_1
where orderid in (select distinct orderid from cte_ar_2)
and AR_severity<>'NA' and AR_severity is not null)
,cte_ar_4 AS (
select *,
case when AR_severity='severe aortic regurgitation' then '1'
when AR_severity='moderate to severe aortic regurgitation' then '2'
when AR_severity='moderate aortic regurgitation' then '3'
when AR_severity='Mild to moderate aortic regurgitation' then '4'
when AR_severity='mild aortic regurgitation' then '5'
when AR_severity='indeterminate' then '6'
end as AR_severity_priority
from cte_ar_3)
,cte_ar_5 AS (
SELECT * FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC,AR_severity_priority ASC
  ) as row_num FROM cte_ar_4 )LOOP WHERE LOOP.row_num=1)
,cte_ar_6 as (
SELECT * FROM cte_ar_1
WHERE ORDERID IN (
select distinct orderid from CTE_ar_1
except
select distinct orderid from CTE_ar_2))
,CTE_ar_7 as (
select distinct orderid,patientid,studydate
from cte_ar_6)
,cte_ar_8 as (
select * from (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC
  ) as row_num FROM CTE_ar_7)loop where loop.row_num=1)
,cte_ar_9 as (
select distinct orderid,patientid,studydate,
case when orderid in (select distinct orderid from cte_ar_8) then 'NA' end as AR_severity
from cte_ar_8
union
select distinct orderid,patientid,studydate,AR_severity
from cte_ar_5)
----TEMPORARY TABBLE FOR AORTIC REGURGITATION
Select * into #tmpar9 from cte_ar_9
DROP TABLE IF EXISTS NASIR.DBO.AORTIC_REGURGITATION_FULLDATASET_PATIENT
SELECT * INTO NASIR.DBO.AORTIC_REGURGITATION_FULLDATASET_PATIENT
FROM #tmpar9


----------PIPELINE FOR AORTIC STENOSIS---------------------------------
;WITH CTE_AS_1 AS (
select *,
CASE when a.Findings_Name like '%aorticvalve%' then
CASE 
												     
                                                   WHEN a.Findings_Value not like  'Aortic valve not well seen. No Doppler evidence of aortic stenosis or regurgitation.'
												   and a.Findings_Value not like 'Aortic valve not well seen. No Doppler evidence of aortic stenosis.'
												   and a.Findings_Value not like '%no aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no aortic stenosis%'
                                                   and a.Findings_Value not like '%no  aortic%'
                                                   and a.Findings_Value not like 'No aortic valve stenosis.'
                                                   and a.Findings_Value not like '%no evidence of aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidence of aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidenc%' then
case
    when a.Findings_Value like '%aortic valve stenosis%'
	or a.Findings_Value like '%aortic stenosis%' then
	case 
     when a.Findings_Value like '%low flow%' 
	 or a.Findings_Value like '%lowflow%' 
	 or a.Findings_Value like '%low-flow%' 
	 or a.Findings_Value like '%low%flow%' then 
					      case
						      when a.Findings_Value like '%normal flow%' then 'normal flow gradient'
							  when a.Findings_Value like '%paradoxical%' then 'paradoxical low flow low gradient' else 'low flow low gradient' end
							   
						  							  			 
else 'NA'
end
else 'NA'
end
else 'NA'
end
end as 'FLOW_GRADIENT',
CASE when a.Findings_Name like '%aorticvalve%' then
CASE 
												    WHEN a.Findings_Value not like  'Aortic valve not well seen. No Doppler evidence of aortic stenosis or regurgitation.'
													and a.Findings_Value not like 'Aortic valve not well seen. No Doppler evidence of aortic stenosis.'
                                                   and a.Findings_Value not like '%no aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no aortic stenosis%'
                                                   and a.Findings_Value not like '%no  aortic%'
                                                   and a.Findings_Value not like 'No aortic valve stenosis.'
                                                   and a.Findings_Value not like '%no evidence of aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidence of aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidenc%'  then
case
    when a.Findings_Value like '%aortic valve stenosis%'
	or a.Findings_Value like '%aortic stenosis%' then
		        case
		            when a.Findings_Value like '%mild%' 
					or a.Findings_Value like '%trace%' 
					or a.Findings_Value like '%trivial%' 
					or a.Findings_Value like '%minimal%' 
					or a.Findings_Value like '%midl%' then
											                                                                                                                                        case
												                                                                                                                                    when a.Findings_Value not like '%moderate%' then 'mild aortic stenosis'
                             
									      
											                                                                                                                                         when a.Findings_Value like '%mild to moderate%' 
																																													     or a.Findings_Value like '%mild-moderate%' 
																																													     or a.Findings_Value like '%mild-mod%' 
																																													     or a.Findings_Value like '%mild-to-moderate%' 
																																													     or a.Findings_Value like '%mild tp moderate%' then 'mild to moderate aortic stenosis' end
																																													                        
			else
	             case
			         when a.Findings_Value like '%moderate%' and a.Findings_Value not like '%mild%' and a.Findings_Value not like '%critical%' and a.Findings_Value not like '%crtical%' and a.Findings_Value not like '%severe%' then 'moderate aortic stenosis' 
			else
				 case
					 when a.Findings_Value like '%severe%' or a.Findings_Value like '%critical%' or a.Findings_Value like '%crtical%'
					     then
							 case 
								  when a.Findings_Value like '%moderate%' or a.Findings_Value like '%mod%' then 'moderate to severe' 
								   else 'severe aortic stenosis' 
								   end			
else 'indeterminate'
end
end
end
else 'NA'
end
else 'NA'
end
else 'NA'
end as 'AS_SEVERITY'
from ECHO_FULLDATASET a)
,cte_AS_2
as(select * from CTE_AS_1
where AS_SEVERITY<>'NA' and AS_SEVERITY is not null)
,cte_AS_3 AS (
select DISTINCT
ORDERID,PatientID,StudyDate,AS_SEVERITY
from CTE_AS_1
where orderid in (select distinct orderid from CTE_AS_2)
and AS_SEVERITY<>'NA' and AS_SEVERITY is not null)
,cte_AS_4 AS (
select *,
case when AS_SEVERITY='severe aortic stenosis' then '1'
when AS_SEVERITY='moderate to severe aortic stenosis' then '2'
when AS_SEVERITY='moderate aortic stenosis' then '3'
when AS_SEVERITY='Mild to moderate aortic stenosis' then '4'
when AS_SEVERITY='mild aortic stenosis' then '5'
when AS_SEVERITY='indeterminate' then '6'
end as AS_severity_priority
from cte_AS_3)
,cte_AS_5 AS (
SELECT * FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC,AS_severity_priority ASC
  ) as row_num FROM cte_AS_4 )LOOP WHERE LOOP.row_num=1)
,CTE_AS_6 as (
SELECT * FROM CTE_AS_1
WHERE ORDERID IN (
select distinct orderid from CTE_AS_1
except
select distinct orderid from cte_AS_2))
,CTE_AS_7 as (
select distinct orderid,patientid,studydate
from CTE_AS_6)
,CTE_AS_8 as (
select * from (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC
  ) as row_num FROM CTE_AS_7)loop where loop.row_num=1)
,CTE_AS_9 as (
select distinct orderid,patientid,studydate,
case when orderid in (select distinct orderid from CTE_AS_8) then 'NA' end as AS_SEVERITY
from CTE_AS_8
union
select distinct orderid,patientid,studydate,AS_SEVERITY
from cte_AS_5)
----TEMPORARY TABLE FOR AORTIC STENOSIS N=175,399
SELECT * INTO #TMPAS9
FROM CTE_AS_9

DROP TABLE IF EXISTS AORTIC_STENOSIS_FULLDATASET_PATIENT
SELECT * INTO AORTIC_STENOSIS_FULLDATASET_PATIENT
FROM #TMPAS9

-----------PIPELINE FOR FLOW GRADIENT-----------------
;WITH CTE_AS_1 AS (
select *,
CASE when a.Findings_Name like '%aorticvalve%' then
CASE 
												     
                                                   WHEN a.Findings_Value not like  'Aortic valve not well seen. No Doppler evidence of aortic stenosis or regurgitation.'
												   and a.Findings_Value not like 'Aortic valve not well seen. No Doppler evidence of aortic stenosis.'
												   and a.Findings_Value not like '%no aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no aortic stenosis%'
                                                   and a.Findings_Value not like '%no  aortic%'
                                                   and a.Findings_Value not like 'No aortic valve stenosis.'
                                                   and a.Findings_Value not like '%no evidence of aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidence of aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidenc%' then
case
    when a.Findings_Value like '%aortic valve stenosis%'
	or a.Findings_Value like '%aortic stenosis%' then
	case 
     when a.Findings_Value like '%low flow%' 
	 or a.Findings_Value like '%lowflow%' 
	 or a.Findings_Value like '%low-flow%' 
	 or a.Findings_Value like '%low%flow%' then 
					      case
						      when a.Findings_Value like '%normal flow%' then 'normal flow gradient'
							  when a.Findings_Value like '%paradoxical%' then 'paradoxical low flow low gradient' else 'low flow low gradient' end
							   
						  							  			 
else 'NA'
end
else 'NA'
end
else 'NA'
end
end as 'FLOW_GRADIENT',
CASE when a.Findings_Name like '%aorticvalve%' then
CASE 
												    WHEN a.Findings_Value not like  'Aortic valve not well seen. No Doppler evidence of aortic stenosis or regurgitation.'
													and a.Findings_Value not like 'Aortic valve not well seen. No Doppler evidence of aortic stenosis.'
                                                   and a.Findings_Value not like '%no aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no aortic stenosis%'
                                                   and a.Findings_Value not like '%no  aortic%'
                                                   and a.Findings_Value not like 'No aortic valve stenosis.'
                                                   and a.Findings_Value not like '%no evidence of aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidence of aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic stenosis%'
                                                   and a.Findings_Value not like '%without aortic valve stenosis%'
                                                   and a.Findings_Value not like '%no evidenc%'  then
case
    when a.Findings_Value like '%aortic valve stenosis%'
	or a.Findings_Value like '%aortic stenosis%' then
		        case
		            when a.Findings_Value like '%mild%' 
					or a.Findings_Value like '%trace%' 
					or a.Findings_Value like '%trivial%' 
					or a.Findings_Value like '%minimal%' 
					or a.Findings_Value like '%midl%' then
											                                                                                                                                        case
												                                                                                                                                    when a.Findings_Value not like '%moderate%' then 'mild aortic stenosis'
                             
									      
											                                                                                                                                         when a.Findings_Value like '%mild to moderate%' 
																																													     or a.Findings_Value like '%mild-moderate%' 
																																													     or a.Findings_Value like '%mild-mod%' 
																																													     or a.Findings_Value like '%mild-to-moderate%' 
																																													     or a.Findings_Value like '%mild tp moderate%' then 'mild to moderate aortic stenosis' end
																																													                        
			else
	             case
			         when a.Findings_Value like '%moderate%' and a.Findings_Value not like '%mild%' and a.Findings_Value not like '%critical%' and a.Findings_Value not like '%crtical%' and a.Findings_Value not like '%severe%' then 'moderate aortic stenosis' 
			else
				 case
					 when a.Findings_Value like '%severe%' or a.Findings_Value like '%critical%' or a.Findings_Value like '%crtical%'
					     then
							 case 
								  when a.Findings_Value like '%moderate%' or a.Findings_Value like '%mod%' then 'moderate to severe' 
								   else 'severe aortic stenosis' 
								   end			
else 'indeterminate'
end
end
end
else 'NA'
end
else 'NA'
end
else 'NA'
end as 'AS_SEVERITY'
from ECHO_FULLDATASET a)
,cte_AS_2_FLOW
as(select * from CTE_AS_1
where FLOW_GRADIENT<>'NA' and FLOW_GRADIENT is not null)
,cte_AS_3_FLOW
AS (
select DISTINCT
ORDERID,PatientID,StudyDate,FLOW_GRADIENT
from CTE_AS_1
where orderid in (select distinct orderid from cte_AS_2_FLOW)
and FLOW_GRADIENT<>'NA' and FLOW_GRADIENT is not null)
,cte_AS_4_FLOW
AS (
select *,
case when FLOW_GRADIENT='paradoxical low flow low gradient' then '1'
when FLOW_GRADIENT='normal flow gradient' then '2'
when FLOW_GRADIENT='low flow low gradient' then '3'
when FLOW_GRADIENT='No Flow gradient' then '4'
end as FLOW_GRADIENT_priority
from cte_AS_3_FLOW)
,cte_AS_5_FLOW
AS (
SELECT * FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC,FLOW_GRADIENT_priority ASC
  ) as row_num FROM cte_AS_4_FLOW )LOOP WHERE LOOP.row_num=1)
,CTE_AS_6_FLOW
as (
SELECT * FROM CTE_AS_1
WHERE ORDERID IN (
select distinct orderid from CTE_AS_1
except
select distinct orderid from cte_AS_2_FLOW))
,CTE_AS_7_FLOW
as (
select distinct orderid,patientid,studydate
from CTE_AS_6_FLOW)
,CTE_AS_8_FLOW
as (
select * from (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC
  ) as row_num FROM CTE_AS_7_FLOW)loop where loop.row_num=1)
,CTE_AS_9_FLOW
as (
select distinct orderid,patientid,studydate,
case when orderid in (select distinct orderid from CTE_AS_8_FLOW) then 'NA' end as FLOW_GRADIENT
from CTE_AS_8_FLOW
union
select distinct orderid,patientid,studydate,FLOW_GRADIENT
from cte_AS_5_FLOW)

---TEMPORARY TABLE FOR FLOW GRADIENT----N=175,399
SELECT * INTO #TMPAS9FLOW
FROM CTE_AS_9_FLOW
DROP TABLE IF EXISTS FLOW_GRADIENT_FULLDATASET_PATIENT
SELECT * INTO FLOW_GRADIENT_FULLDATASET_PATIENT
FROM #TMPAS9FLOW

------------------------------PIEPLINE FOR DIASTOLIC FUNCTION ---------------------------------------------
;with cte_df_1 as(
select*,
case
    when a.Findings_Value like '%Normal diastolic function%' or a.Findings_Value like '%Grade I%' then
	case
	    when a.Findings_Value like 'Normal diastolic function and LV filling pressures.' 
		                  or a.Findings_Value like 'Normal diastolic function adjusted for age; normal LV filling pressures.' 
			              or a.findings_value like 'Normal diastolic function.' 
			              or a.findings_value like 'Normal diastolic function; normal LV filling pressures.' 
				          or a.findings_value like 'Normal diastolic function for age.'
					      or a.findings_value like 'LV filling pressure is within normal range. Grade I diastolic dysfunction.'
					      or a.findings_value like 'Normal diastolic function adjusted for age, normal LV filling pressures.'
						  or a.findings_value like '%normal diastolic function%'
						  or a.findings_value like '%low-normal diastolic function%'
						    then 'NORMAL'
		  when a.findings_value like 'Diastolic dysfunction Grade I (Mild): Impaired relaxation with normal LV filling pressures.'
		                        or a.findings_value like '%grade I (mild)%'
			                    or a.findings_value like '%grade I:%'
				                or a.findings_value like '%grade I diastolic dysfunction%'
								then 'Grade I'
		  when a.findings_value like 'Diastolic dysfunction Grade II (Moderate): Impaired relaxation with elevated LV filling pressures'
		                         or a.findings_value like '%Grade II (Moderate)%'
								 or a.findings_value like '%grade II diastolic%'
								 then 'Grade II'
		  when a.findings_value like 'Diastolic dysfunction Grade III (Severe): Impaired relaxation with restrictive LV filling pressures.'
		                         or findings_value like '%grade III %'
								 then 'Grade III'
		  when a.findings_value like 'Unable to assess diastolic function.'
		                          or a.findings_value like 'Unable to assess diastolic function due to Tachycardia.'
								  or a.findings_value like 'Unable to assess diastolic function due to suboptimal quality of recording.'
								  or a.findings_value like 'Unable to assess diastolic function due to mitral inflow obstruction.'
								  or a.findings_value like 'Unable to assess diastolic function due to severe MAC.'
								  or a.findings_value like '%unable to%'
								  then 'indeterminate'
			else 'NA'
		end
			end as 'df_severity'
from ECHO_FULLDATASET a)
,cte_df_2
as(
select * from cte_df_1
where df_severity<>'NA' and df_severity is not null)
,cte_df_3 AS (
select DISTINCT
ORDERID,PatientID,StudyDate,df_severity
from cte_df_1
where orderid in (select distinct orderid from cte_df_2)
and df_severity<>'NA' and df_severity is not null)
,cte_df_4 AS (
select *,
case when df_severity='Grade III' then '1'
when df_severity='Grade II' then '2'
when df_severity='Grade I' then '3'
when df_severity='NORMAL' then '4'
when df_severity='indeterminate' then '5'
end as df_severity_priority
from cte_df_3)
,cte_df_5 as (
SELECT * FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC,df_severity_priority ASC
  ) as row_num FROM cte_df_4 )LOOP WHERE LOOP.row_num=1)
,cte_df_6 as (
SELECT * FROM cte_df_1
WHERE ORDERID IN (
select distinct orderid from cte_df_1
except
select distinct orderid from cte_df_2))
,cte_df_7
as (
select distinct orderid,patientid,studydate
from cte_df_6)
,cte_df_8 as (
select * from (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY orderid ORDER BY studydate DESC
  ) as row_num FROM cte_df_7)loop where loop.row_num=1)
 ,cte_df_9 as (
select distinct orderid,patientid,studydate,
case when orderid in (select distinct orderid from cte_df_8) then 'NA' end as df_severity
from cte_df_8
union
select distinct orderid,patientid,studydate,df_severity
from cte_df_5)

---TEMPORARY TABLE FOR DIASTOLIC FUNCTION---N=175,399
Select * into #tmpdf9
from cte_df_9
DROP TABLE IF EXISTS DIASTOLIC_FUNCTION_FULLDATASET_PATIENT
SELECT * INTO DIASTOLIC_FUNCTION_FULLDATASET_PATIENT
FROM #tmpdf9


------------------------PIPELINE FOR MITACLIP---PROSTHETIC CLIP FOR MITRAL VALVES------------------------------
;WITH CTE_MP_1 AS (select *,
CASE when a.Findings_Name like '%mitralvalve%' then
     CASE 
	     WHEN
             a.Findings_Value  like '%mitraclip%' 
	         OR a.Findings_Value like '%mitraclip%'
	         OR a.Findings_Value like '%clip%' then '1' else '0' 
			 END
			 end as MV_MITRACLIP,
CASE when a.Findings_Name like '%mitralvalve%' then
     CASE 
	     WHEN
	          a.Findings_Value like '%Prosthetic%' 
                     OR  a.Findings_Value like '%Bioprosthetic%'
	                 OR  a.Findings_Value like '%Paravalvular%'
		             OR  a.Findings_Value like '%paravalvualr%'
		             OR a.Findings_Value like '%paravalvuar%'
		             OR a.Findings_Value like '%paravalvualr%'
		             OR a.Findings_Value like '%Paraval%'
		             OR a.Findings_Value like '%Transvalvular%'
					 OR a.Findings_Value like '%mechanical%' 
					 OR a.Findings_Value like '%surgical ring%'
					 OR a.Findings_Value like '%tavr%'  then '1' else '0' 
					 END
					 end as MV_PROSTHETIC
from ECHO_FULLDATASET a)

--------------------------FINDING THE TOTAL NUMBER OF ORDERID WITH MITRA CLIP AND PROSTHETIC COMING ONLY FROM MITRAL VALVE  FINDING COLUMNS NAME-----
,cte_MP_2 as(
select * from CTE_MP_1 a
where a.orderid in
(
select distinct orderid from CTE_MP_1 where MV_MITRACLIP=1
union
select distinct orderid from CTE_MP_1 where MV_PROSTHETIC=1
)
and 
a.Findings_Name like '%mitralvalve%')

----------------------------FINDINGS MAX COUNT AT ORDERID LEVEL----
,cte_MP_3 as (
select distinct 
orderid,max(MV_MITRACLIP) MV_MITRACLIP,max(MV_PROSTHETIC) MV_PROSTHETIC from cte_MP_2
group by OrderID)

---LABELLING THE EXCLUSIVE SET THAT DOES NOT CONTAIN MITRA CLIP OR PROSTHETIC----
,cte_MP_4 as (
select distinct orderid,
case
   when orderid in (select distinct orderid from CTE_MP_1 except (select distinct orderid from cte_MP_3)) then '0' else '1' end as MV_MITRACLIP,
case
   when orderid in (select distinct orderid from CTE_MP_1 except (select distinct orderid from cte_MP_3)) then '0' else '1' end as MV_PROSTHETIC
from (select distinct orderid from CTE_MP_1
      except 
      (select distinct orderid from cte_MP_3))a)

------------------MERGING TWO SETS-----
,cte_MP_5 as (
select * from cte_MP_4
union
select * from cte_MP_3)
,cte_MP_6 as (
select distinct a.*,b.PatientID,b.StudyDate from cte_MP_5 a
left join ECHO_FULLDATASET b 
on a.OrderID=b.OrderID)
---TEMPORARY TABLE FOR MITRALVALVE N=175,399
select * into #tmpmitral from cte_MP_6
DROP TABLE IF EXISTS MITRALVALVE_FULLDATASET_PATIENT
SELECT * INTO MITRALVALVE_FULLDATASET_PATIENT
FROM #tmpmitral

------AORTIC VALVE------MITRAL VALVE---PROSTHETIC VALVE--------
;WITH CTE_AV_1 AS (select *,
CASE when a.Findings_Name like '%aorticvalve%'  then
     CASE 
	     WHEN
             a.Findings_Value  like '%mitraclip%' 
	         OR a.Findings_Value like '%mitraclip%'
	         OR a.Findings_Value like '%clip%' then '1' else '0' 
			 END
			 end as AORTICVALVE_MITRACLIP,
CASE when a.Findings_Name like '%aorticvalve%' then
     CASE 
	     WHEN
	          a.Findings_Value like '%Prosthetic%' 
                     OR  a.Findings_Value like '%Bioprosthetic%'
	                 OR  a.Findings_Value like '%Paravalvular%'
		             OR  a.Findings_Value like '%paravalvualr%'
		             OR a.Findings_Value like '%paravalvuar%'
		             OR a.Findings_Value like '%paravalvualr%'
		             OR a.Findings_Value like '%Paraval%'
		             OR a.Findings_Value like '%Transvalvular%'
					 OR a.Findings_Value like '%mechanical%' 
					 OR a.Findings_Value like '%surgical ring%'
					 OR a.Findings_Value like '%tavr%'  then '1' else '0' 
					 END
					 end as AORTICVALVE_PROSTHETIC
from ECHO_FULLDATASET a)
---FINDING THE TOTAL NUMBER OF ORDERID WITH MITRA CLIP AND PROSTHETIC COMING ONLY FROM MITRAL VALVE  FINDING COLUMNS NAME-----
,CTE_AV_2 as(
select * from CTE_AV_1 a
where a.orderid in
(
select distinct orderid from CTE_AV_1 where AORTICVALVE_MITRACLIP=1
union
select distinct orderid from CTE_AV_1 where AORTICVALVE_PROSTHETIC=1
)
and 
a.Findings_Name like '%aorticvalve%')
---FINDINGS MAX COUNT AT ORDERID LEVEL----
,cte_AV_3 as (
select distinct 
orderid,max(AORTICVALVE_MITRACLIP) AORTICVALVE_MITRACLIP,max(AORTICVALVE_PROSTHETIC) AORTICVALVE_PROSTHETIC from CTE_AV_2
group by OrderID)
------LABELLING THE EXCLUSIVE SET THAT DOES NOT CONTAIN MITRA CLIP OR PROSTHETIC----
,cte_AV_4 as (
select distinct orderid,
case
   when orderid in (select distinct orderid from CTE_AV_1 except (select distinct orderid from cte_AV_3)) then '0' else '1' end as AORTICVALVE_MITRACLIP,
case
   when orderid in (select distinct orderid from CTE_AV_1 except (select distinct orderid from cte_AV_3)) then '0' else '1' end as AORTICVALVE_PROSTHETIC
from (select distinct orderid from CTE_AV_1
      except 
      (select distinct orderid from cte_AV_3))a)
---MERGING TWO SETS-----
,cte_AV_5 as (
select * from cte_AV_4
union
select * from cte_AV_3)
,cte_AV_6 as (
select distinct a.*,b.PatientID,b.StudyDate from cte_AV_5 a
left join ECHO_FULLDATASET b 
on a.OrderID=b.OrderID)
---TEMPORARY TABLE FOR AORTIC VALVE N=175,399
select * into #tmpaortic
from cte_AV_6
DROP TABLE IF EXISTS AORTICVALVE_FULLDATASET_PATIENT
SELECT * INTO AORTICVALVE_FULLDATASET_PATIENT
FROM #tmpaortic

---PIPELINE FOR EJECTION FRACTION----------------------
----EJECTION FRACTION----------------------
;with CTE_EJECTION_1  
as  (select distinct patientid,orderid,StudyDate,Findings_Name,
findings_value from NASIR.DBO.ECHO_FULLDATASET
where [Findings_Value] LIKE '%LV EF%'
OR [Findings_Value] LIKE '%LVEF%'
OR [Findings_Value] LIKE '% LVEF %'
OR [Findings_Value] LIKE '% LV EF %'
OR [Findings_Value] LIKE '% EF %'
OR [Findings_Value] LIKE '%left%ventricular%ejection%fraction%'
OR [Findings_Value] LIKE '%ejection%fraction%'
OR [Findings_Value] LIKE '%left%ventricular%systolic%fraction%'
OR [Findings_Value] LIKE '%ejection%fraction%'
OR [Findings_Value] LIKE '%systolic%ventric%'
OR [Findings_Value] LIKE '%cardiac%function%'
OR [Findings_Value] LIKE '%ventricular%function%'
OR [Findings_Value] LIKE '%ventricular contractility%'
OR [Findings_Value] LIKE '%systolic contractility%'
OR [Findings_Value] LIKE '%overall contractility%'
OR [Findings_Value] LIKE '%ventricular contraction%'
OR [Findings_Value] LIKE '%systolic performance%'
OR [Findings_Value] LIKE '%ejection%fraction%'
OR [Findings_Value] LIKE '%ejection% fraction%'
OR [Findings_Value] LIKE '% ejection%fraction %'
OR [Findings_Value] LIKE '% ejection %fraction %'
OR [Findings_Value] LIKE '%EF = %'
OR [Findings_Value] LIKE '%EF=%'
OR [Findings_Value] LIKE '%EF is%'
OR [Findings_Value] LIKE '%EF by%'
OR [Findings_Value] LIKE '%EF varies with RR interval from%'
OR [Findings_Value] LIKE '%EF calculated by 3D%'
OR [Findings_Value] LIKE '%EF ranges from%'
OR [Findings_Value] LIKE '%EF =%'
OR [Findings_Value] LIKE '%EF varies with RR from%'
OR [Findings_Value] LIKE '%EF varies between%'
OR [Findings_Value] LIKE '%EF [0-9]%'
)

SELECT * INTO #TMPRAWEF FROM CTE_EJECTION_1 
;WITH CTE_EJECTION_2 AS
(
select distinct* from #TMPRAWEF where findings_value like '%[0-9]%')
,CTE_EJECTION_3 
as
(select patientid,orderid,studydate,findings_value,case
when CHARINDEX('.',findings_value)>0 then SUBSTRING(findings_value,1,CHARINDEX('.',findings_value)-1)
when CHARINDEX(',',findings_value)>0 then SUBSTRING(findings_value,1,CHARINDEX(',',findings_value)-1)
when CHARINDEX(';',findings_value)>0 then SUBSTRING(findings_value,1,CHARINDEX(';',findings_value)-1) 
         else findings_value end firstpart,
    CASE 
	   WHEN CHARINDEX('.',findings_value)>0 THEN SUBSTRING(findings_value,CHARINDEX('.',findings_value)+1,len(findings_value))
	   WHEN CHARINDEX(',',findings_value)>0 THEN SUBSTRING(findings_value,CHARINDEX(',',findings_value)+1,len(findings_value))
	   WHEN CHARINDEX(';',findings_value)>0 THEN SUBSTRING(findings_value,CHARINDEX(';',findings_value)+1,len(findings_value))
         ELSE NULL END as lastpart
from CTE_EJECTION_2)
,CTE_EJECTION_4 
as(
select * from (
select patientid,orderid,studydate,findings_value,firstpart as partEF
from CTE_EJECTION_3 where firstpart is not null and firstpart <> ''
union
select patientid,orderid,studydate,findings_value,lastpart as partEF
from CTE_EJECTION_3 where lastpart is not null and lastpart <> '')abc
where abc.partEF is not null)
,CTE_EJECTION_5 
as (select patientid,orderid,studydate,
findings_value,partEF from CTE_EJECTION_4 where partEF like '%[0-9]%')
,CTE_EJECTION_6 
as (
select * from CTE_EJECTION_5 where partEF like '%LV EF%'
or partEF like '%estimated EF%'
or partEF like '%estimated%'
or partEF like '% EF %'
or partEF like '%LVEF%'
or partEF like '%ejection fraction%'
or partEF LIKE '%LV EF%'
or partEF LIKE '%LVEF%'
or partEF LIKE '% LVEF %'
or partEF LIKE '% LV EF %'
or partEF LIKE '% EF %'
or partEF LIKE '%EF = %'
or partEF LIKE '%EF=%'
or partEF LIKE '%EF is%'
OR partEF LIKE '%EF=%'
OR partEF LIKE '%EF by%'
OR partEF like '%EF varies with RR interval from%'
OR partEF like '%EF calculated by 3D%'
OR partEF like '%EF ranges from%'
OR partEF like '%EF =%'
OR partEF like '%EF varies with RR from%'
OR partEF like '%EF varies between%'
OR partEF like '%EF [0-9]%'
)
,CTE_EJECTION_7
as (
select patientid,orderid,studydate,
findings_value,partEF,
substring(partEF, PatIndex('%[0-9]%', partEF), len(partEF)) as EF_probable from CTE_EJECTION_6)
,CTE_EJECTION_8 
as (
select *,
case 
	when EF_probable like '%[0-9]%-%[0-9]%' then replace(replace(EF_probable,' - ',LTRIM(RTRIM(' - '))),'%','')
    when EF_probable like '%[0-9]%to%[0-9]%' then replace(replace(EF_probable,' to ',LTRIM(RTRIM(' - '))),'%','')
	when EF_probable LIKE '3D=%' and EF_probable not like '3D = 178 mL, ESV= 102 mL, LV EF by 3D= 43%.' then replace(SUBSTRING(EF_probable, 4,5), '%','')
	when EF_probable LIKE '3D = %' and EF_probable not like '3D = 178 mL, ESV= 102 mL, LV EF by 3D= 43%.' then replace(SUBSTRING(EF_probable, 6,7), '%','')
	when EF_probable LIKE '3D =%' and EF_probable not like '3D = 178 mL, ESV= 102 mL, LV EF by 3D= 43%.' then replace(SUBSTRING(EF_probable, 5,6), '%','')
	when EF_probable LIKE '3D LV ejection fraction=%' THEN replace(SUBSTRING(EF_probable, 25,26), '%','')
	WHEN EF_probable LIKE '3D calculated EF =%' THEN replace(SUBSTRING(EF_probable, 19,20), '%','')
	WHEN EF_probable LIKE '3D calculated LVEF is%' THEN replace(SUBSTRING(EF_probable, 23,24), '%','')
	WHEN EF_probable LIKE '3D EF is%' THEN replace(SUBSTRING(EF_probable, 10,11), '%','')
	WHEN EF_probable LIKE '3D is%' and EF_probable not like '3d is appx%' and EF_probable not like '3d is normal at%' and EF_probable not like '3D is performed by%' THEN replace(SUBSTRING(EF_probable, 6,7), '%','')
	WHEN EF_probable LIKE '3D LVEF =%' THEN replace(SUBSTRING(EF_probable, 10,11), '%','') 
	WHEN EF_probable LIKE '3D LVEF%' AND EF_probable NOT LIKE '3D LVEF =%' AND EF_probable NOT LIKE '3D LVEF is%' AND EF_probable NOT LIKE '%global%' AND EF_probable NOT LIKE '%data%' THEN trim(replace(SUBSTRING(EF_probable, 9,10), '%',''))
	WHEN EF_probable LIKE '3D LVEF ~%' AND EF_probable NOT LIKE '3D LVEF =%' AND EF_probable NOT LIKE '3D LVEF is%' AND EF_probable NOT LIKE '%global%' AND EF_probable NOT LIKE '%data%' THEN trim(replace(SUBSTRING(EF_probable, 10,11), '%',''))
	WHEN EF_probable LIKE '3D LVEF is%' THEN trim(replace(SUBSTRING(EF_probable, 12,13), '%',''))
	WHEN EF_probable LIKE '3D LV ejection fraction%' THEN trim(replace(SUBSTRING(EF_probable, 25,26), '%',''))
	WHEN EF_probable LIKE '3D measurement is%' THEN trim(replace(SUBSTRING(EF_probable, 18,19), '%',''))
	WHEN EF_probable LIKE '3D method is%' THEN trim(replace(SUBSTRING(EF_probable, 13,14), '%',''))
	WHEN EF_probable LIKE '3D quantitation is%' THEN trim(replace(SUBSTRING(EF_probable, 19,20), '%',''))
	WHEN EF_probable LIKE '3D volume =%' THEN trim(replace(SUBSTRING(EF_probable, 11,12), '%',''))
	WHEN EF_probable LIKE '3D volumes is%' THEN trim(replace(SUBSTRING(EF_probable, 14,15), '%',''))
	WHEN EF_probable LIKE '3D EF is%' then replace(SUBSTRING(EF_probable, 10, 11),'%','')
	WHEN EF_probable LIKE '3D is%'  and EF_probable not like '3D is normal%' then replace(SUBSTRING(EF_probable, 7, 8),'%','')
	WHEN EF_probable LIKE '3D is 62% LV GLS is normal at -21%' then '62'
	WHEN EF_probable LIKE '3D LV ejection fraction=%' then replace(SUBSTRING(EF_probable, 25, 26),'%','')
	WHEN EF_probable LIKE '3 D LV ejection fraction=%' then replace(SUBSTRING(EF_probable, 26, 27),'%','')
	WHEN EF_probable LIKE '3D LVEF=%' then replace(SUBSTRING(EF_probable, 10, 11),'%','')
	WHEN EF_probable LIKE '3D LV EF =%' then replace(SUBSTRING(EF_probable, 12, 13),'%','')
	WHEN EF_probable LIKE '3D LVEF =%' then replace(SUBSTRING(EF_probable, 10, 11),'%','')
	WHEN EF_probable LIKE '3 D LVEF ~ %' then replace(SUBSTRING(EF_probable, 12, 13),'%','')
	WHEN EF_probable LIKE '3 D LVEF= %' then replace(SUBSTRING(EF_probable, 11, 12),'%','')
	WHEN EF_probable LIKE '3D LVEF [0-9][0-9]%' then replace(SUBSTRING(EF_probable, 9, 10),'%','')
	WHEN EF_probable LIKE '3D LVEF is [0-9][0-9]%' then replace(SUBSTRING(EF_probable, 12, 13),'%','')
	WHEN EF_probable LIKE '3D measurement is%' then replace(SUBSTRING(EF_probable, 18, 19),'%','')
	WHEN EF_probable LIKE '3D method is%' then replace(SUBSTRING(EF_probable, 14, 15),'%','')
	WHEN EF_probable LIKE '3D quantitation is%' then replace(SUBSTRING(EF_probable, 19,20),'%','')
	WHEN EF_probable LIKE '3D volume =%' then replace(SUBSTRING(EF_probable, 12,13),'%','')
	WHEN EF_probable LIKE '3D volumes is%' then replace(SUBSTRING(EF_probable, 15,16),'%','')
	WHEN EF_probable LIKE '3D volume ejection fraction= %' then replace(SUBSTRING(EF_probable, 30,31),'%','')
	WHEN EF_probable LIKE '3D=%' and EF_probable not like '3D = 178 mL, ESV= 102 mL, LV EF by 3D= 43%.' then replace(SUBSTRING(EF_probable,5,6),'%','')
	WHEN EF_probable LIKE '3D calculated EF =%' then replace(SUBSTRING(EF_probable,19,20),'%','')
	WHEN EF_probable LIKE '3D calculated LVEF is%' then replace(SUBSTRING(EF_probable,22,23),'%','')
	WHEN EF_probable LIKE '3 D LVEF=%' then replace(SUBSTRING(EF_probable,10,11),'%','')
	WHEN EF_probable LIKE '3 D LVEF=appx %' then replace(SUBSTRING(EF_probable,15,16),'%','')
	WHEN EF_probable LIKE '%3D is appx%' then replace(SUBSTRING(EF_probable,12,13),'%','')
	WHEN EF_probable LIKE '3D derived EF = %' then replace(SUBSTRING(EF_probable,17,18),'%','')
	WHEN EF_probable LIKE '3D derived EF %' then replace(SUBSTRING(EF_probable,15,16),'%','')
	WHEN EF_probable LIKE '3D derived LVEF = %' then replace(SUBSTRING(EF_probable,19,20),'%','')
	WHEN EF_probable LIKE '3D derived LVEF is %' then replace(SUBSTRING(EF_probable,20,21),'%','')
	WHEN EF_probable LIKE '3 D LVEF =%' then replace(SUBSTRING(EF_probable,11,12),'%','')
	WHEN EF_probable LIKE '3d volume measurements is %' then replace(SUBSTRING(EF_probable,27,28),'%','')
	WHEN EF_probable LIKE '3D, RV EDV = 152 mL, ESV= 79 mL, RV EF = 48.1%.' then '48'
	WHEN EF_probable LIKE '114 mL, RV ESV= 61.5 mL, RV SV= 53 mL, RV EF = 46%.' then '46'
	WHEN EF_probable LIKE '3D EF = %' then replace(SUBSTRING(EF_probable,9,10),'%','')
	WHEN EF_probable LIKE '3D EF=%' then replace(SUBSTRING(EF_probable,7,8),'%','')
	WHEN EF_probable LIKE '3D is normal at%' then replace(SUBSTRING(EF_probable,17,18),'%','')
	WHEN EF_probable LIKE '%3D = 178 mL, ESV= 102 mL, LV EF by 3D= 43%.%' then '43'
	WHEN EF_probable LIKE '9%. 3D LVEF is 66%.' THEN '66'
	WHEN EF_probable LIKE ' Estimated at%' then replace(SUBSTRING(EF_probable,15,19),'%','')
	WHEN EF_probable LIKE ' EF=%' then replace(SUBSTRING(EF_probable,7,8),'%','')
	WHEN EF_probable LIKE 'EF is%' then replace(SUBSTRING(EF_probable,7,8),'%','')
	WHEN EF_probable LIKE 'EF by 3D =%' then replace(SUBSTRING(EF_probable,11,12),'%','')
	WHEN EF_probable LIKE 'EF varies with RR interval from [0-9]-[0-9]%' then replace(SUBSTRING(EF_probable,33,37),'%','')
	WHEN EF_probable LIKE 'EF varies with RR interval from [0-9] to [0-9]%' then replace(SUBSTRING(EF_probable,33,40),'%','')
	WHEN EF_probable LIKE 'EF calculated by 3D is%' then replace(SUBSTRING(EF_probable,24,25),'%','')
	WHEN EF_probable LIKE 'EF ranges from%' then replace(SUBSTRING(EF_probable,16,22),'%','')
	WHEN EF_probable LIKE 'EF =%' then replace(SUBSTRING(EF_probable,5,6),'%','')
	WHEN EF_probable LIKE 'EF varies with RR from%' then replace(SUBSTRING(EF_probable,24,31),'%','')
	WHEN EF_probable LIKE 'EF varies between%' then replace(SUBSTRING(EF_probable,19,24),'%','')
	WHEN EF_probable LIKE 'EF [0-9]%' then replace(SUBSTRING(EF_probable,4,5),'%','')
			else SUBSTRING(EF_probable,
PATINDEX('%[0-9]%', EF_probable),
(CASE WHEN PATINDEX('%[^0-9]%', STUFF(EF_probable, 1, (PATINDEX('%[0-9]%', EF_probable) - 1), '')) = 0
THEN LEN(EF_probable) ELSE (PATINDEX('%[^0-9]%', STUFF(EF_probable, 1, (PATINDEX('%[0-9]%', EF_probable) - 1), ''))) - 1
END ))
end AS ExtractString
from CTE_EJECTION_7)
,CTE_EJECTION_9 as(
select *,DBO.GetNumbers(ExtractString) as semi_ExtractString
from CTE_EJECTION_8) 
,CTE_EJECTION_10 
AS (
select *,
substring(semi_ExtractString, 1, len(semi_ExtractString)+1-patindex('%[^. ]%', reverse(semi_ExtractString))) as FINAL_EF
from CTE_EJECTION_9)
,CTE_EJECTION_11 AS (
select distinct patientid,orderid,StudyDate,findings_value,FINAL_EF,
case
   when FINAL_EF like '%[0-9]-[0-9]%' then '2'
   when FINAL_EF like '%[0-9]--[0-9]%' then '2'
   WHEN FINAL_EF LIKE '%[0-9]%' AND FINAL_EF NOT LIKE '%-%' THEN '1' END AS DATA_TYPE
from CTE_EJECTION_10)
SELECT *
INTO #TMPEJECTION
FROM CTE_EJECTION_11
;WITH CTE_EJECTION_PRE12 AS (
SELECT * FROM #TMPEJECTION WHERE FINAL_EF NOT IN ('100','110','120','130','140','150','160','170','180','190','200','2020','3500'))
,CTE_EJECTION_12 AS (
SELECT *,
  ROW_NUMBER() OVER(PARTITION BY patientid ORDER BY studydate desc,DATA_TYPE asc) as row_num
FROM CTE_EJECTION_PRE12)
SELECT 
* INTO #TMPEJECTION_1
FROM CTE_EJECTION_12 
--WHERE row_num=1 (AT ORDERID LEVEL (THIS WILL BE DONE FOR ORDERID LEVEL))
;WITH CTE_EJECTION_13 AS (
SELECT * FROM #TMPRAWEF WHERE PatientID IN (
SELECT DISTINCT PATIENTID FROM #TMPRAWEF
EXCEPT (SELECT DISTINCT PATIENTID FROM #TMPEJECTION_1)))
,CTE_EJECTION_14 AS(
SELECT DISTINCT patientid,orderid,studydate,FINDINGS_VALUE  FROM (
SELECT DISTINCT patientid,orderid,studydate,FINDINGS_VALUE 
FROM CTE_EJECTION_13 WHERE Findings_Value LIKE '%[0-9]%')ABC
WHERE ABC.Findings_Value LIKE '%ESTIMATE%' AND ABC.Findings_Value NOT LIKE '%UNDERESTIMATE%')
,CTE_EJECTION_15
as
(select patientid,orderid,studydate,findings_value,
case
when CHARINDEX('.',findings_value)>0 then SUBSTRING(findings_value,1,CHARINDEX('.',findings_value)-1)
when CHARINDEX(',',findings_value)>0 then SUBSTRING(findings_value,1,CHARINDEX(',',findings_value)-1)
when CHARINDEX(';',findings_value)>0 then SUBSTRING(findings_value,1,CHARINDEX(';',findings_value)-1) 
         else findings_value end firstpart,
    CASE 
	   WHEN CHARINDEX('.',findings_value)>0 THEN SUBSTRING(findings_value,CHARINDEX('.',findings_value)+1,len(findings_value))
	   WHEN CHARINDEX(',',findings_value)>0 THEN SUBSTRING(findings_value,CHARINDEX(',',findings_value)+1,len(findings_value))
	   WHEN CHARINDEX(';',findings_value)>0 THEN SUBSTRING(findings_value,CHARINDEX(';',findings_value)+1,len(findings_value))
         ELSE NULL END as lastpart
from CTE_EJECTION_14)
,CTE_EJECTION_16 
as(
select * from (
select patientid,orderid,studydate,findings_value,firstpart as partEF
from CTE_EJECTION_15 where firstpart is not null and firstpart <> ''
union
select patientid,orderid,studydate,findings_value,lastpart as partEF
from CTE_EJECTION_15 where lastpart is not null and lastpart <> '')abc
where abc.partEF is not null)
,CTE_EJECTION_17 
as (select patientid,orderid,studydate,findings_value,
partEF from CTE_EJECTION_16 where partEF like '%[0-9]%')
,CTE_EJECTION_18
as (
select patientid,orderid,studydate,findings_value,partEF,
substring(partEF, PatIndex('%[0-9]%', partEF), len(partEF)) as EF_probable from CTE_EJECTION_17)
,CTE_EJECTION_19 as(
select *,DBO.GetNumbers(EF_probable) as semi_ExtractString
from CTE_EJECTION_18)
,CTE_EJECTION_20 AS (
SELECT *,
case
WHEN Findings_Value LIKE '%around 50%' THEN '50' 
	WHEN Findings_Value like '%severe%' and Findings_Value not like '%moderate%' then 16 
    WHEN Findings_Value like '%moderate%severe%' then 30 
    WHEN Findings_Value like '%mild%moderate%' then 45 
    WHEN Findings_Value like '%poor%' then 45  
    WHEN Findings_Value like '%sluggish%' then 45 
    WHEN Findings_Value like '%reduce%' then 45 
    WHEN Findings_Value like '%decrease%' then 45 
    WHEN Findings_Value like '%depress%' then 45 
    WHEN Findings_Value like '%impair%' then 45 
    WHEN Findings_Value like '%abnormal%' then 45 
    WHEN Findings_Value like '%below normal%' then 45 
    WHEN Findings_Value like '%normal%' then 55 
    WHEN Findings_Value like '%no abnormal%' then 55 
    WHEN Findings_Value like '%intact%' then 55 
    WHEN Findings_Value like '%preserved%' then 55  
    WHEN Findings_Value like '%adequate%' then 55 
    WHEN Findings_Value like '%good%' then 55 
    WHEN Findings_Value like '%satisfactory%' then 55  
    WHEN Findings_Value like '%excellent%' then 55 
    WHEN Findings_Value like '%hyperdynamic%' then 70 
    WHEN Findings_Value like '%hyperkinetic%' then 70 
    WHEN Findings_Value like  '%vigorous%' then 70 end as EF_text,
case
WHEN Findings_Value LIKE '%around 50%' THEN '3'
	WHEN Findings_Value like '%severe%' THEN '3' 
    WHEN Findings_Value like '%moderate%severe%' THEN '3'
    WHEN Findings_Value like '%mild%moderate%' THEN '3'
    WHEN Findings_Value like '%poor%' THEN '3'  
    WHEN Findings_Value like '%sluggish%' THEN '3'
    WHEN Findings_Value like '%reduce%' THEN '3'
    WHEN Findings_Value like '%decrease%' THEN '3'
    WHEN Findings_Value like '%depress%' THEN '3' 
    WHEN Findings_Value like '%impair%' THEN '3'
    WHEN Findings_Value like '%abnormal%' THEN '3'
    WHEN Findings_Value like '%below normal%' THEN '3' 
    WHEN Findings_Value like '%normal%' THEN '3'
    WHEN Findings_Value like '%no abnormal%' THEN '3' 
    WHEN Findings_Value like '%intact%' THEN '3' 
    WHEN Findings_Value like '%preserved%' THEN '3'  
    WHEN Findings_Value like '%adequate%' THEN '3' 
    WHEN Findings_Value like '%good%' THEN '3' 
    WHEN Findings_Value like '%satisfactory%' THEN '3'  
    WHEN Findings_Value like '%excellent%' THEN '3' 
    WHEN Findings_Value like '%hyperdynamic%' THEN '3' 
    WHEN Findings_Value like '%hyperkinetic%' THEN '3' 
    WHEN Findings_Value like  '%vigorous%' THEN '3' end as DATA_TYPE
FROM CTE_EJECTION_13
where orderid not in (select distinct orderid from cte_ejection_19))
SELECT * INTO #TMPEFTEXT
FROM CTE_EJECTION_20

;WITH CTE_EJECTION_21 AS (
SELECT DISTINCT PATIENTID,ORDERID,StudyDate,FINDINGS_VALUE,CAST(FINAL_EF AS VARCHAR) FINAL_EF,DATA_TYPE
FROM #TMPEJECTION_1
UNION ALL
SELECT DISTINCT PATIENTID,ORDERID,StudyDate,FINDINGS_VALUE,CAST(EF_TEXT AS VARCHAR) AS FINAL_EF,DATA_TYPE
FROM #TMPEFTEXT)
,CTE_EJECTION_22 AS (
SELECT *,
  ROW_NUMBER() OVER(PARTITION BY patientid ORDER BY studydate desc,DATA_TYPE asc) as row_num
FROM CTE_EJECTION_21)
,CTE_EJECTION_23 AS (
SELECT PATIENTID,ORDERID,StudyDate,FINDINGS_VALUE,FINAL_EF,DATA_TYPE FROM CTE_EJECTION_22 
--WHERE row_num=1 (THIS WILL BE DONE FOR ORDERID LEVEL)
)
SELECT * INTO #TMPFINALEJECTION
FROM CTE_EJECTION_23 WHERE DATA_TYPE IS NOT NULL

DROP TABLE IF EXISTS EJECTIONFRACTION_FULLDATASET_PATIENT
SELECT * INTO EJECTIONFRACTION_FULLDATASET_PATIENT
FROM #TMPFINALEJECTION

;WITH CTE_EF_1 AS (SELECT *,
CASE
WHEN FINDINGS_VALUE LIKE '%3D EF performed but with suboptimal traCKING%' THEN '0'
WHEN FINDINGS_VALUE LIKE '%3D EF=60%%' THEN '60'
WHEN FINDINGS_VALUE LIKE '%3D EF=62%%' THEN '62'
WHEN FINDINGS_VALUE LIKE '%LV EF by 3D is 62%%' THEN '62'
WHEN FINDINGS_VALUE LIKE '%3D EF=63%%' THEN '63'
WHEN FINDINGS_VALUE LIKE '%3D EF=68%%' THEN '68'
WHEN FINDINGS_VALUE LIKE '%3D EF=73%%' THEN '73'
WHEN FINDINGS_VALUE LIKE '%3D EF=67%%' THEN '67'
WHEN FINDINGS_VALUE LIKE '%3D LVEF 67%%' THEN '67'
WHEN FINDINGS_VALUE LIKE '%EF = 8%%' THEN '8'
WHEN FINDINGS_VALUE LIKE '%Estimated EF is >70%%' THEN '70'
WHEN FINDINGS_VALUE LIKE '%Estimated EF is 25-29%%' THEN '27'
WHEN FINDINGS_VALUE LIKE '%Overall wall motion is normal. 3D EF=51%%' THEN '51'
WHEN FINDINGS_VALUE LIKE '%RV size is normal. By 3D, RV EDV= 139.6 mL%%' THEN '52'
WHEN FINDINGS_VALUE LIKE '%The average global longitudinal strain is normal at-21.4%. 3D EF=63%%' THEN '63'
WHEN FINDINGS_VALUE LIKE '%3D EF = 57.7%.%' THEN '57' END AS FINALF_EF
FROM EJECTIONFRACTION_FULLDATASET_PATIENT WHERE DATA_TYPE='1'
AND FINAL_EF IN ('1','2','3','4','5','57.7','8'))

,CTE_EF_POINT AS (SELECT [PATIENTID],[ORDERID],[StudyDate],[FINDINGS_VALUE],FINALF_EF AS FINAL_EF,DATA_TYPE FROM CTE_EF_1 WHERE FINALF_EF IS NOT NULL
UNION
SELECT [PATIENTID],[ORDERID],[StudyDate],[FINDINGS_VALUE],FINAL_EF,DATA_TYPE FROM CTE_EF_1 WHERE FINALF_EF IS NULL
UNION
SELECT [PATIENTID],[ORDERID],[StudyDate],[FINDINGS_VALUE],FINAL_EF,DATA_TYPE FROM [NASIR].[dbo].EJECTIONFRACTION_FULLDATASET_PATIENT WHERE DATA_TYPE='1' AND FINAL_EF NOT IN ('1','2','3','4','5','57.7','8'))

,CTE_EF_POINT_FINAL AS (SELECT * FROM CTE_EF_POINT
UNION
SELECT [PATIENTID],[ORDERID],[StudyDate],[FINDINGS_VALUE],FINAL_EF,DATA_TYPE FROM [NASIR].[dbo].EJECTIONFRACTION_FULLDATASET_PATIENT WHERE DATA_TYPE NOT IN ('1'))

SELECT * INTO #TMPEF
FROM CTE_EF_POINT_FINAL

DROP TABLE IF EXISTS EJECTIONFRACTION_FULLDATASET_PATIENT
SELECT * INTO EJECTIONFRACTION_FULLDATASET_PATIENT FROM #TMPEF

------------------------------------------------------------------
CODE TO GENERATE THE MEAN FOR BANDS 
-----------------------------------------------------------------

;WITH CTE_BANDS_CONV AS (select *,
    case when CHARINDEX('-',FINAL_EF)>0 
         then SUBSTRING(FINAL_EF,1,CHARINDEX('-',FINAL_EF)-1) 
         else FINAL_EF end LB, 
    CASE WHEN CHARINDEX('-',FINAL_EF)>0 
         THEN SUBSTRING(FINAL_EF,CHARINDEX('-',FINAL_EF)+1,len(FINAL_EF))  
         ELSE NULL END as UB
from EJECTIONFRACTION_FULLDATASET_PATIENT WHERE DATA_TYPE=2)

,CTE_CONV2 AS (SELECT PATIENTID,ORDERID,StudyDate,Findings_Value,DATA_TYPE,
FINAL_EF,
CAST(LB AS INT) LB,
CAST(UB AS INT) UB
FROM CTE_BANDS_CONV)

,CTE_CONV3 AS (SELECT *, (LB+UB)/2 MEAN_EF FROM CTE_CONV2)


,CTE_CONV4
AS
(SELECT PATIENTID,ORDERID,StudyDate,Findings_Value,DATA_TYPE,FINAL_EF,MEAN_EF AS FINALEF FROM CTE_CONV3
UNION
SELECT PATIENTID,ORDERID,StudyDate,Findings_Value,DATA_TYPE,FINAL_EF,CAST(FINAL_EF AS float) FINALEF  
FROM NASIR.DBO.EJECTIONFRACTION_FULLDATASET_PATIENT
WHERE DATA_TYPE IN ('1','3'))


SELECT * INTO #TMPFCONVEF
FROM CTE_CONV4

;WITH CTE_CONV5 AS (SELECT *,
CASE 
WHEN FINALEF>=0 AND FINALEF<=10 THEN '0-10'
WHEN FINALEF>=11 AND FINALEF<=20 THEN '11-20'
WHEN FINALEF>=21 AND FINALEF<=30 THEN '21-30'
WHEN FINALEF>=31 AND FINALEF<=40 THEN '31-40'
WHEN FINALEF>=41 AND FINALEF<=50 THEN '41-50'
WHEN FINALEF>=51 AND FINALEF<=60 THEN '51-60'
WHEN FINALEF>=61 AND FINALEF<=70 THEN '61-70'
WHEN FINALEF>=71 AND FINALEF<=80 THEN '71-80'
WHEN FINALEF>=81 AND FINALEF<=90 THEN '81-90'
WHEN FINALEF>=91 AND FINALEF<=100 THEN '91-100' END AS NEWBANDS
FROM #TMPFCONVEF)

SELECT * INTO #TMPFCONVEF2
FROM CTE_CONV5 WHERE NEWBANDS IS NOT NULL

DROP TABLE IF EXISTS EJECTIONFRACTION_FULLDATASET_PATIENT
SELECT * INTO EJECTIONFRACTION_FULLDATASET_PATIENT
FROM #TMPFCONVEF2

DROP INDEX [IX_ECHOP] ON NASIR.DBO.EJECTIONFRACTION_FULLDATASET_PATIENT
CREATE NONCLUSTERED INDEX [IX_ECHOP] ON NASIR.DBO.EJECTIONFRACTION_FULLDATASET_PATIENT
	(
		PatientID,ORDERID ASC
	)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF,
	DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]


---CREATING QUALITATIVE FINAL ECHO OUTPUT TABLE
SELECT 
[PATIENTID],[ORDERID],[STUDYDATE]
  INTO #TMPUNIONALLIDS
FROM [NASIR].[dbo].[MITRAL_REGURGITATION_FULLDATASET_PATIENT]

UNION
(SELECT [PATIENTID],[ORDERID],[STUDYDATE]
FROM [NASIR].[dbo].AORTIC_REGURGITATION_FULLDATASET_PATIENT)

UNION
(SELECT [PATIENTID],[ORDERID],[STUDYDATE]
FROM[NASIR].[dbo].AORTIC_STENOSIS_FULLDATASET_PATIENT)

UNION
(SELECT [PATIENTID],[ORDERID],[STUDYDATE]
FROM[NASIR].[dbo].MITRAL_STENOSIS_FULLDATASET_PATIENT)

UNION

(SELECT [PATIENTID],[ORDERID],[STUDYDATE]
FROM [NASIR].[dbo].FLOW_GRADIENT_FULLDATASET_PATIENT)

UNION

(SELECT [PATIENTID],[ORDERID],[STUDYDATE]
FROM [NASIR].[dbo].DIASTOLIC_FUNCTION_FULLDATASET_PATIENT)

UNION
(SELECT [PATIENTID],[ORDERID],[STUDYDATE]
FROM [NASIR].[dbo].EJECTIONFRACTION_FULLDATASET_PATIENT)

UNION 
(SELECT [PATIENTID],[ORDERID],[STUDYDATE] 
FROM [dbo].[MITRALVALVE_FULLDATASET_PATIENT])

UNION
(SELECT [PATIENTID],[ORDERID],[STUDYDATE] 
FROM [dbo].[AORTIC_STENOSIS_FULLDATASET_PATIENT])


SELECT a.*,
b.MR_SEVERITY,
c.MS_SEVERITY,
e.AR_SEVERITY,
d.AS_SEVERITY,
f.FLOW_GRADIENT,
g.DF_SEVERITY,
i.[DATA_TYPE] as EF_DATATYPE,i.[FINAL_EF],i.[FINALEF],i.[NEWBANDS],
j.[MV_MITRACLIP],j.[MV_PROSTHETIC],
k.AORTICVALVE_MITRACLIP,k.AORTICVALVE_PROSTHETIC
into #tmpalldata
FROM #TMPUNIONALLIDS a
LEFT JOIN [dbo].[MITRAL_REGURGITATION_FULLDATASET_PATIENT] b
ON a.PATIENTID=b.PATIENTID
AND a.ORDERID=b.ORDERID
AND a.STUDYDATE=b.STUDYDATE 

LEFT JOIN [dbo].MITRAL_STENOSIS_FULLDATASET_PATIENT c
ON a.PATIENTID=c.PATIENTID
AND a.ORDERID=c.ORDERID
AND a.STUDYDATE=c.STUDYDATE 

LEFT JOIN [dbo].AORTIC_STENOSIS_FULLDATASET_PATIENT d
ON a.PATIENTID=d.PATIENTID
AND a.ORDERID=d.ORDERID
AND a.STUDYDATE=d.STUDYDATE

LEFT JOIN AORTIC_REGURGITATION_FULLDATASET_PATIENT e
ON a.PATIENTID=e.PATIENTID
AND a.ORDERID=e.ORDERID
AND a.STUDYDATE=e.STUDYDATE

LEFT JOIN [dbo].[FLOW_GRADIENT_FULLDATASET_PATIENT] f
ON a.PATIENTID=f.PATIENTID
AND a.ORDERID=f.ORDERID
AND a.STUDYDATE=f.STUDYDATE

LEFT JOIN [dbo].[DIASTOLIC_FUNCTION_FULLDATASET_PATIENT] g
ON a.PATIENTID=g.PATIENTID
AND a.ORDERID=g.ORDERID
AND a.STUDYDATE=g.STUDYDATE

LEFT JOIN [dbo].[EJECTIONFRACTION_FULLDATASET_PATIENT] i
ON a.PATIENTID=i.PATIENTID
AND a.ORDERID=i.ORDERID
AND a.STUDYDATE=i.STUDYDATE

LEFT JOIN MITRALVALVE_FULLDATASET_PATIENT j
ON a.PATIENTID=j.PATIENTID
AND a.ORDERID=j.ORDERID
AND a.STUDYDATE=j.STUDYDATE

LEFT JOIN AORTICVALVE_FULLDATASET_PATIENT k
ON a.PATIENTID=k.PATIENTID
AND a.ORDERID=k.ORDERID
AND a.STUDYDATE=k.STUDYDATE

DROP TABLE IF EXISTS NASIR.DBO.ECHO_QUALITATIVE_OUTPUT
SELECT * INTO NASIR.DBO.ECHO_QUALITATIVE_OUTPUT
FROM #tmpalldata


DROP TABLE #TMPFCONVEF2
DROP TABLE #TMPFCONVEF
DROP TABLE #TMPEF
DROP TABLE #TMPFINALEJECTION
DROP TABLE #TMPEJECTION
DROP TABLE #TMPEFTEXT
DROP TABLE #TMPEJECTION_1
DROP TABLE #TMPRAWEF
DROP TABLE #tmpaortic
DROP TABLE #tmpmitral
DROP TABLE #tmpdf9
DROP TABLE #TMPAS9FLOW
DROP TABLE #tmpar9
DROP TABLE #TMPAS9
DROP TABLE #tmpmss9
DROP TABLE #tmpmr9
DROP TABLE #tmpalldata
