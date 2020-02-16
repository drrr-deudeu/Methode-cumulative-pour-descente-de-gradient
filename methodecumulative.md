Définitions :

$$Soit~ n \in \N,~ tel~ que~ 1<n$$ 
$$\forall~l'ensemble~\{X\}_{n} =\{x_{1},...,x_{i},...,x_{n}\} ~, dans~\R$$
$$On~définit~\overline{x}_{(n)} ~la~moyenne~des~n~éléments~x_{i}~telle~que:$$
$$\overline{x}_{(n)} = \frac{1}{n}\sum_{i=1}^{n}x_i~~~~(a)$$
$$Notons~au~passage~qu'on~a~donc~:$$
$$n\overline{x}_{(n)}=\sum_{i=1}^{n}x_i~~~~(b)$$
$$On~définit~maintenant~\sigma_{(n)}(x) ~l'écart-type~de~n~éléments~x_{i}~et~la~variance~V_{(n)}(x)~tels~que:$$
$$\sigma_{(n)}(x) = \sqrt{V_{(n)}(x)} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\overline{x})^2}$$
$$On~peut~écrire~:$$
$$V_{(n)}(x)=\frac{1}{n}\sum_{i=1}^{n}(x_{i}^{2}-2\overline{x}x_{i}+\overline{x}^{2})$$
$$V_{(n)}(x)=\frac{1}{n}(\sum_{i=1}^{n}x_{i}^{2}-2\overline{x}\sum_{i=1}^{n}x_{i}+\sum_{i=1}^{n}\overline{x}^{2})$$
$$V_{(n)}(x)=\frac{1}{n}(\sum_{i=1}^{n}x_{i}^{2}-2n\overline{x}\overline{x}+n\overline{x}^{2})$$
$$Donc~V_{n}(x)=\frac{1}{n}\sum_{i=1}^{n}x_{i}^{2}-\overline{x}^{2}_{(n)}~~~~(c)$$

$$Soit~maintenant~m \in \N,~ tel~ que~ 1<m$$
$$Soit~l'ensemble~\{X\}_{n+m} =\{x_{1},...,x_{i},...,x_{n},x_{n+1},...,x_{n+m}\} ~dans~\R$$
$$Alors~la~moyenne~des~n+m~éléments~x_{i}~,~\overline{x}_{(n+m)}:$$
$$\overline{x}_{(n+m)} = \frac{1}{n+m}\sum_{i=1}^{n+m}x_i$$
$$\overline{x}_{(n+m)} = \frac{1}{n+m}(\sum_{i=1}^{n}x_i+\sum_{i=n+1}^{m}x_i)$$
$$en~utilisant~(b)~on~a~:$$
$$\overline{x}_{(n+m)} = \frac{1}{n+m}(n\overline{x}_{(n)}+m\overline{x}_{(m)})~~~~(c)$$
$$Calculons~maintenant~la~variance~de~l' ensemble~(n+m)$$
$$V_{(n+m)}(x) = \frac{1}{n+m}\sum_{i=1}^{n+m}(x_i-\overline{x})^2$$
$$V_{(n+m)}(x) = \frac{1}{n+m}\sum_{i=1}^{n+m}x_i^{2}-\overline{x}_{(n+m)}^{2}$$
$$V_{(n+m)}(x) = \frac{1}{n+m}(\sum_{i=1}^{n}x_i^{2}+\sum_{i=n+1}^{m}x_i^{2})-\overline{x}_{(n+m)}^{2}$$

$$On~définit~la~normalisation~de~l'ensemble~{X_{(n)}},~la~transformation ~faisant~passer~x_{i}~à~X_{i}~comme~suit~:$$
$$X_{i}=\frac{x_{i}-\overline{x}}{\sigma(x)}~~~~(d)$$
$$Notons~ au~ passage~ le~ résultat~ suivant~ :$$
$$\overline{X_{(n)}}=\frac{1}{n\sigma_{(n)}(x)}\sum_{i=1}^{n}(x_i-\overline{x})=0~~~~(e)$$
$$Méthode~de~ la~ descente~ de~ gradient~ appliquée~ à~ une~ droite~ de~ régression~ linéaire~ telle~ que~:$$
$$\hat{Y} = F(X) = \theta_0+\theta_1X~~~~(e)$$

$$La~méthode~consiste~à~s'approcher~itérativement~des~valeurs~optimales~ des~\theta~$$
$$i.e.~celles~pour~lesquelles~le~coût~est~minimum~.~Les~ équations$$
$$proposées~ pour~ résoudre~ ce~ problème~ ne~ sont~ valables~ qu'avec~ des~ données~ normalisées~ :$$

$$Chaque~itération~donne~des~nouvelles~valeurs~des~\theta :$$
$$\theta_0 \leftarrow \theta_0 - d\theta_0$$
$$\theta_1 \leftarrow \theta_1 - d\theta_1$$
$$Avec~:$$
$$d\theta_0=\frac{\alpha}{n}\sum_{i=1}^{n}(\hat{Y_i}-Y_i) ~ ~et~~d\theta_1=\frac{\alpha}{n}\sum_{i=1}^{n}(\hat{Y_i}-Y_i)X_i$$
$$avec~ \alpha~ définit~ comme~ étant~ le~ "learning~ rate",~$$
$$ \hat{Y_i}~la~ valeur~ prédite~ en~ X_i~ et~ Y_i~ la~ valeur~ normalisée~ de~ y_i$$
$$On~ peut~ donc~ écrire~:$$
$$\frac{d\theta_0}{\alpha}=\frac{1}{n}\sum_{i=1}^{n}\hat{Y_i}-\overline{Y}_{(n)}=\frac{1}{n}\sum_{i=1}^{n}(\theta_0+\theta_1X_i)-\overline{Y}_{(n)}$$
$$En~ appliquant~ (e)~:$$
$$\frac{d\theta_0}{\alpha}=\theta_0$$
$$On~ a~ donc~ à~ chaque~ itération~:$$
$$\theta_0 \leftarrow \theta_0-{\alpha}\theta_0$$

$$De~ même ~:$$
$$\frac{d\theta_1}{\alpha}=\frac{1}{n}\sum_{i=1}^{n}(\hat{Y_i}-\overline{Y}_{(n)})X_i$$
$$Mais~ d'après~ (e),~ on~ a~ \overline{Y}_{(n)}=0,~d'où~:$$
$$\frac{d\theta_1}{\alpha}=\frac{1}{n}\sum_{i=1}^{n}X_iY_i$$
$$Donc~ \theta_1\leftarrow\theta_1-\frac{\alpha}{n}\sum_{i=1}^{n}X_iY_i$$
$$Pour~ l'ensemble~ (n+m)~ on~ aura~ donc~ :$$
$$\theta_0 \leftarrow \theta_0-{\alpha}\theta_0$$
$$\theta_1\leftarrow\theta_1-\frac{\alpha}{n+m}(\sum_{i=1}^{n}X_iY_i+\sum_{i=n+1}^{n+m}X_iY_i)$$

$$Enfin~ la~ dénormalisation~ des~ \theta~:$$
$$D'après~ (d) et (f),~ on~ a~$$
$$\frac{\hat{y}_{i}-\overline{y}_{(n)}}{\sigma_{(n)}(y)} = \theta_0 - \theta_1\frac{x_{i}-\overline{x}_{(n)}}{\sigma_{(n)}(x)}$$
$$\hat{y}_{i}=\sigma_{(n)}(y)(\theta_0-\frac{theta_1}{\sigma_{(n)}(x)}\overline{x}_{(n)}) + \overline{y}_{(n)}+\theta_1\frac{\sigma_{(n)}(y)}{\sigma_{(n)}(x)}x_i$$
$$D'où,~par~identification~:$$
$$THETA(0) = \sigma_{(n)}(y)(\theta_0-\frac{\theta_1}{\sigma_{(n)}(x)}\overline{x}_{(n)}) + \overline{y}_{(n)}$$
$$THETA(1) = \theta_1\frac{\sigma_{(n)}(y)}{\sigma_{(n)}(x)}$$

$$Conclusion:$$
$$Le~calcul~ des~ \theta~ pour~ un~ ensemble~ n+m~,~ ne~ nécessitent~ que~ la~ connaissance~ des~ grandeurs~ suivantes~:$$
$$-~ Les~ \theta~ sur~ l'ensemble~ X_{(n)}$$
$$-~Les~ grandeurs: \overline{x}_{(n)},\overline{y}_{(n)},\sum_{i=1}^{n}x_i^{2},\sum_{i=1}^{n}y_i^{2},\sum_{i=1}^{n}X_iY_i$$
$$-~ l'ensemble~ des~ \{X\}_{(m)}~ et~ des~ \{Y\}_{(m)}$$

Cas particulier:
$$Avec~ l'ajoût~ d'une~ seule~ valeur~x_{n+1},~on~aura~:$$
$$\overline{x}_{(n+1)}=\frac{1}{n+1}(n\overline{x}_{(n)}+x_{n+1})$$
$$V_{(n+1)}(x) = \frac{1}{n+m}(\sum_{i=1}^{n}x_i^{2}+x_{n+1}^{2})-\overline{x}_{(n+m)}^{2}$$
$$\sigma_{(n+1)}(x) = \sqrt{V_{(n+1)}}$$
$$X_{n+1}=\frac{x_{n+1}-\overline{x}_{(n+1)}}{\sigma_{(n+1)}(x)}$$
