select  survived, round(AVG(age)::numeric, 1) as age_moyen, count(*) as total
from passagers 
where age is not null 
group by  survived ;

select pclass, round(AVG(fare)::numeric, 2) as prix_moyen, round(MIN(fare)::numeric, 2) as prix_min, round(MAX(fare)::numeric, 2) as prix_max
from passagers 
group by pclass 
order by pclass;

-- valeurs manquantes
select 
	count(*) - count(age) as age_manquant,
	count(*) - count(fare) as prix_manquants,
	count(*) - count(embarked) as embarked_manquant
from passagers ;
