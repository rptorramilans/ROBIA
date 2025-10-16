# ROBIA
**ROBIA:** The spatial distribution of a dye tracer is evaluated from **UAV-based RGB imagery**, processed through a robust and systematic approach developed to detect rhodamine dye presence and estimate its concentration at the water surface.

![Flow figure](https://github.com/user-attachments/assets/28f38521-5558-4867-8f01-e00354c5dae8)

The workflow starts with UAV image acquisition and subsequently extraction of **RGB pixel values**. From these, a set of combinatory indices (based on the RGB values) is computed to estimate rhodamine intensity in the water surface. In-situ measured dye concentrations are then linked to the **UAV-derived indices** through regression analysis, selecting the best index according to the coefficient of determination (R2) and root mean square error (RMSE). The method also incorporates **physical constraints**: coastlines, riverbanks and also artificial structures, such as aquaculture farms, by applying a masking process to exclude their influence on rhodamine mapping. The approach was tested in a rhodamine survey in **Fangar Bay (NW Mediterranean Sea)**, a small and shallow coastal bay with aquaculture rafts.

This application has demonstrated the importance of integrating dye tracing with infrastructure identification for accurate mapping results. The method is suitable for rhodamine experiments in obstructed environments such as marinas, offshore wind farms and aquaculture facilities. The code and example datasets are openly available, and can easily be adapted to dispersion studies in diverse aquatic environments.

**Code developer:** Raquel Peñas Torramilans from Laboratori d'Enginyeria Marítima (LIM-UPC), Universitat Politècnica de Catalunya (UPC-BarcelonaTech), **raquel.penas@upc.edu**

**Cite** ...

<img width="1229" height="922" alt="image1107" src="https://github.com/user-attachments/assets/fbd71f75-60e6-4651-8741-d22510721aa7" />
