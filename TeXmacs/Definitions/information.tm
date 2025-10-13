<TeXmacs|2.1.4>

<style|generic>

<\body>
  <section|Definitions Informations Theory>

  <subsection|Entropy>

  <subsubsection|Von Neumann Entropy>

  <\equation>
    S<around*|(|\<rho\>|)>=S<rsub|\<rho\>>=-Tr<around*|(|\<rho\>*log<around*|(|\<rho\>|)>|)>=<big|sum><rsub|i>\<lambda\><rsub|i>*log<around*|(|<frac|1|\<lambda\><rsub|i>>|)>
  </equation>

  Trick:

  <\equation*>
    S<around*|(|\<rho\>|)>=-*lim<rsub|r\<rightarrow\>1><frac|\<partial\>|\<partial\>r>Tr<around*|(|\<rho\><rsup|r>|)>
  </equation*>

  <\proof>
    <with|color|red|add evaluate lines! ><math|<frac|\<partial\>|\<partial\>r>\<rho\><rsup|r><around*|\||<rsub|r\<equallim\>1>|\<nobracket\>>=<frac|\<partial\>|\<partial\>r>e<rsup|log<around*|(|\<rho\>|)>r><around*|\||<rsub|r\<equallim\>1>|\<nobracket\>>=log<around*|(|\<rho\>|)>*e<rsup|log<around*|(|\<rho\>|)>r><around*|\||<rsub|r\<equallim\>1>|\<nobracket\>>=log<around*|(|\<rho\>|)>*\<rho\><rsup|r><around*|\||<rsub|r\<equallim\>1>|\<nobracket\>>>
  </proof>

  <subsubsection|Reni Entropy>

  <subsection|Entropic Quantities>

  <subsubsection|Conditional Entropy>

  <em|Conditional Entropy> quantifies the amount of information needed to
  describe outcome of a random variable X given another random variable Y is
  known.

  <\equation>
    S<rsub|X<around*|\||Y|\<nobracket\>>>=S<rsub|XY>-S<rsub|Y>
  </equation>

  The conditional entropy is zero <math|S<rsub|X<around*|\||Y|\<nobracket\>>>=0>
  if the state of the <math|X> is completly known, given the information of
  system Y.\ 

  <subsubsection|Coherent Information>

  <subsubsection|Mutual Information>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-2|<tuple|1.1|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-3|<tuple|1.1.1|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-4|<tuple|1.1.2|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-5|<tuple|1.2|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-6|<tuple|1.2.1|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-7|<tuple|1.2.2|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-8|<tuple|1.2.3|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Definitions
      Informations Theory> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Entropy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|2tab>|1.1.1<space|2spc>Von Neumann Entropy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|2tab>|1.1.2<space|2spc>Reni Entropy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Entropic Quantities
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|1.2.1<space|2spc>Mutual Information
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|2tab>|1.2.2<space|2spc>Conditional Entropy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|2tab>|1.2.3<space|2spc>Coherent Information
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>
    </associate>
  </collection>
</auxiliary>