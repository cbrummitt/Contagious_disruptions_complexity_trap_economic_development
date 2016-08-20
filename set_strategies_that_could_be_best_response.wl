(* ::Package:: *)

(* ::Section:: *)
(*Set of strategies that could be a best response*)


(* ::Text:: *)
(*This function strategiesThatCouldBeBestResponsesFaster computes the set of strategies {(Subscript[m, 1],Subscript[\[Tau], 1]), ...} that could potentially be a best response:*)


(* ::Input::Initialization:: *)
strategiesThatCouldBeBestResponses[{\[Alpha]_,\[Beta]_},F_]/;0<=F<=1&&\[Alpha]>0&&0<=\[Beta]<=1:=
Which[
(* first line of Equation SI-11 *)
F==0,
{{0,0}}, 

(* second line of Equation SI-11 *)
F==1,
{{Ceiling[(\[Alpha]/\[Beta])^(1/(\[Beta]-1))],Ceiling[(\[Alpha]/\[Beta])^(1/(\[Beta]-1))]},{Floor[(\[Alpha]/\[Beta])^(1/(\[Beta]-1))],Floor[(\[Alpha]/\[Beta])^(1/(\[Beta]-1))]}}, 

(* third line of Equation SI-11 *)
True,
Module[{mMax},
(* from Corollary 1, the best response degree m^* < \[Alpha]^(-1/(1-\[Beta])) *)
mMax=Ceiling[\[Alpha]^(-1/(1-\[Beta]))]; 

Cases[Flatten[Table[{m,\[Tau]},{m,0,mMax},{\[Tau],0,m}],1],{m_,\[Tau]_}/;
(
(* Equation SI-10 *)
(m==\[Tau]&&m<((-1+\[Beta]) ProductLog[((1/\[Alpha])^(1/(1-\[Beta])) Log[F])/(-1+\[Beta])])/Log[F]) 

(* last line of Equation SI-11 *)
||(m>\[Tau]&&m<\[Tau]^\[Beta]/\[Alpha])
)]]
]


strategiesThatCouldBeBestResponses::usage = 
"strategiesThatCouldBeBestResponses[{\[Alpha], \[Beta]}, F] computes the set of strategies {{\!\(\*SubscriptBox[\(m\), \(1\)]\), \!\(\*SubscriptBox[\(\[Tau]\), \(1\)]\)}, {\!\(\*SubscriptBox[\(m\), \(2\)]\), \!\(\*SubscriptBox[\(\[Tau]\), \(2\)]\)}, ...} that could potentially be a best response.";


(* ::Section:: *)
(*Chance of successfully producing*)


probSuccess[{m_, \[Tau]_},F_]=
If[{m, \[Tau]} =!= {0,0},
	Probability[S >= \[Tau], S \[Distributed] BinomialDistribution[m, F]],
0];

probSuccess::usage = "probSuccess[{m, \[Tau]}, F] computes the probability that a binomial random variable with parameters m (number of trials) and F (chance of success) is at least \[Tau].";
